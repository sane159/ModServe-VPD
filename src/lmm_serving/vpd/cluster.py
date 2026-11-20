import io
import os
import queue
import threading
import time
import traceback
from typing import Any, Dict, Generator, List, Optional
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmm_serving.vpd.model_stages import (
    run_prefill_stage,
    run_vision_stage,
    prune_model_for_stage,
    unpack_kv_cache,
)
from lmm_serving.vpd.scheduler import RouteLease, VPDRouter
from lmm_serving.vpd.transport import VPDTransport
from lmm_serving.vpd.vision_cache import VisionFeatureLRU, encode_images_with_cache, try_full_hit_fast_path


def _worker_entry(
    rank: int,
    world_size: int,
    role: str,
    model_id: str,
    data_backend: str,
    master_port: int,
    v_task_queue,
    event_queue,
    enable_vision_cache: bool = True,
    vision_cache_max_bytes: int = 256 * 1024 * 1024,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["HF_HUB_OFFLINE"] = "1"

    device = f"cuda:{rank}" if data_backend == "nccl" else "cuda:0"
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        transport = VPDTransport(data_backend=data_backend)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device).eval()
        eos_id = processor.tokenizer.eos_token_id
        before_mb, after_mb = prune_model_for_stage(model, role)
        print(f"[{role} rank={rank}] weights pruned: {before_mb:.1f}MB -> {after_mb:.1f}MB", flush=True)
        dist.barrier()
        event_queue.put({"type": "READY", "rank": rank, "role": role})

        if role == "V":
            v_cache = VisionFeatureLRU(max_bytes=vision_cache_max_bytes) if enable_vision_cache else None
            while True:
                task = v_task_queue.get()
                if task.get("type") == "STOP":
                    transport.send_control({"type": "STOP"}, dst_rank=task["p_rank"])
                    break
                if task.get("inject_crash"):
                    raise RuntimeError(f"Injected worker crash on rank {rank} for req {task['req_id']}")

                req_id = task["req_id"]
                messages = [
                    {
                        "role": "user",
                        "content": ([{"type": "image"} for _ in task["images_bytes"]] + [{"type": "text", "text": task["prompt"]}]),
                    }
                ]
                prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

                fast = try_full_hit_fast_path(processor, prompt_text, task["images_bytes"], v_cache)
                if fast is not None:
                    input_ids_cpu, attn_mask_cpu, vision_feats = fast
                else:
                    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in task["images_bytes"]]
                    inputs = processor(text=prompt_text, images=images, return_tensors="pt").to(device)
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                    vision_feats = encode_images_with_cache(
                        model, processor, images, task["images_bytes"],
                        inputs["pixel_values"], inputs["pixel_attention_mask"],
                        v_cache, isolate_by_req_shape=True, pad_to_orig_batch=True,
                    )
                    input_ids_cpu = inputs["input_ids"].cpu()
                    attn_mask_cpu = inputs["attention_mask"].cpu()

                meta = {
                    "req_id": req_id,
                    "dst_decode": task["d_rank"],
                    "input_ids": input_ids_cpu.tolist(),
                    "attention_mask": attn_mask_cpu.tolist(),
                    "max_new_tokens": task["max_new_tokens"],
                }
                transport.send_tensor_with_ack(vision_feats, meta=meta, dst_rank=task["p_rank"])
                del vision_feats
                # V 在收到 P 的交接 ACK 后立即释放槽位
                event_queue.put({"type": "STAGE_RELEASE", "req_id": req_id, "stage": "V", "rank": rank})

        elif role == "P":
            while True:
                header, vision_feats = transport.recv_tensor_with_ack(src_rank=0, target_device=device)
                if header.get("type") == "STOP":
                    for d_rank in (2, 3):
                        transport.send_control({"type": "STOP"}, dst_rank=d_rank)
                    break

                req_id = header["req_id"]
                dst_decode = header["dst_decode"]
                input_ids = torch.tensor(header["input_ids"], dtype=torch.long, device=device)
                attention_mask = torch.tensor(header["attention_mask"], dtype=torch.long, device=device)

                first_token, packed_kv = run_prefill_stage(model, input_ids, attention_mask, vision_feats)
                del vision_feats

                first_tid = int(first_token.item())
                first_text = processor.tokenizer.decode([first_tid], skip_special_tokens=True)
                # Prefill 算出首 Token 立刻发往网关（seq_idx=0），最大化降低 TTFT
                event_queue.put(
                    {
                        "type": "TOKEN",
                        "req_id": req_id,
                        "seq_idx": 0,
                        "token_id": first_tid,
                        "text": first_text,
                        "rank": rank,
                    }
                )

                meta = {
                    "req_id": req_id,
                    "first_token": first_token.cpu().tolist(),
                    "attention_mask": header["attention_mask"],
                    "max_new_tokens": header["max_new_tokens"],
                }
                transport.send_tensor_with_ack(packed_kv, meta=meta, dst_rank=dst_decode)
                del packed_kv
                # P 在收到 D 的交接 ACK 后立即释放槽位
                event_queue.put({"type": "STAGE_RELEASE", "req_id": req_id, "stage": "P", "rank": rank})

        elif role == "D":
            while True:
                header, packed_kv = transport.recv_tensor_with_ack(src_rank=1, target_device=device)
                if header.get("type") == "STOP":
                    break

                req_id = header["req_id"]
                first_token = torch.tensor(header["first_token"], dtype=torch.long, device=device)
                attention_mask = torch.tensor(header["attention_mask"], dtype=torch.long, device=device)
                max_new_tokens = header["max_new_tokens"]

                past_key_values = unpack_kv_cache(packed_kv)
                del packed_kv
                cur_token = first_token
                cur_mask = attention_mask
                emitted = 1

                with torch.inference_mode():
                    for step in range(1, max_new_tokens):
                        if eos_id is not None and int(cur_token.item()) == eos_id:
                            break
                        cur_mask = torch.cat(
                            [cur_mask, torch.ones((cur_mask.shape[0], 1), device=device, dtype=cur_mask.dtype)],
                            dim=1,
                        )
                        outputs = model(
                            input_ids=cur_token,
                            attention_mask=cur_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        cur_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        past_key_values = outputs.past_key_values
                        tid = int(cur_token.item())
                        text = processor.tokenizer.decode([tid], skip_special_tokens=True)
                        event_queue.put(
                            {
                                "type": "TOKEN",
                                "req_id": req_id,
                                "seq_idx": step,
                                "token_id": tid,
                                "text": text,
                                "rank": rank,
                            }
                        )
                        emitted += 1

                event_queue.put({"type": "DONE", "req_id": req_id, "total_tokens": emitted, "d_rank": rank})
                # D 在生成结束后释放槽位
                event_queue.put({"type": "STAGE_RELEASE", "req_id": req_id, "stage": "D", "rank": rank})

        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        event_queue.put(
            {
                "type": "WORKER_ERROR",
                "rank": rank,
                "role": role,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        )


class _RequestStream:
    """按 req_id 重排 Token，保证 seq_idx 严格递增输出"""

    def __init__(self, req_id: str):
        self.req_id = req_id
        self.out_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.next_seq = 0
        self.pending: Dict[int, Dict[str, Any]] = {}
        self.done_event: Optional[Dict[str, Any]] = None

    def push_token(self, ev: Dict[str, Any]) -> None:
        self.pending[ev["seq_idx"]] = ev
        self._flush()

    def mark_done(self, ev: Dict[str, Any]) -> None:
        self.done_event = ev
        self._flush()

    def _flush(self) -> None:
        while self.next_seq in self.pending:
            self.out_q.put(self.pending.pop(self.next_seq))
            self.next_seq += 1
        if self.done_event is not None and self.next_seq >= self.done_event["total_tokens"]:
            self.out_q.put(self.done_event)
            self.done_event = None

    def fail(self, error_msg: str) -> None:
        self.out_q.put({"type": "ERROR", "req_id": self.req_id, "error": error_msg})


class VPDCluster:
    """管理 1V+1P+2D worker 生命周期、Token 分发重排与异常整组回收"""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        data_backend: str = "gloo",
        master_port: int = 29521,
        max_admitted: int = 4,
        enable_vision_cache: bool = True,
        vision_cache_max_bytes: int = 256 * 1024 * 1024,
    ):
        self.model_id = model_id
        self.data_backend = data_backend
        self.master_port = master_port
        self.enable_vision_cache = enable_vision_cache
        self.vision_cache_max_bytes = vision_cache_max_bytes
        self.ctx = mp.get_context("spawn")
        self.v_task_queue = self.ctx.Queue()
        self.event_queue = self.ctx.Queue()

        self.router = VPDRouter(
            vision_ranks=[0],
            prefill_ranks=[1],
            decode_ranks=[2, 3],
            v_slots=1,
            p_slots=1,
            d_slots=1,
            max_admitted=max_admitted,
        )
        self.procs: List[mp.Process] = []
        self._streams: Dict[str, _RequestStream] = {}
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._failed_reason: Optional[str] = None
        self._dispatcher: Optional[threading.Thread] = None

    def start(self) -> None:
        roles = [(0, "V"), (1, "P"), (2, "D"), (3, "D")]
        for rank, role in roles:
            p = self.ctx.Process(
                target=_worker_entry,
                args=(
                    rank,
                    4,
                    role,
                    self.model_id,
                    self.data_backend,
                    self.master_port,
                    self.v_task_queue if role == "V" else None,
                    self.event_queue,
                    self.enable_vision_cache,
                    self.vision_cache_max_bytes,
                ),
            )
            p.start()
            self.procs.append(p)

        ready = 0
        while ready < 4:
            ev = self.event_queue.get(timeout=120)
            if ev["type"] == "READY":
                ready += 1
            elif ev["type"] == "WORKER_ERROR":
                self._abort_cluster(ev["error"])
                raise RuntimeError(f"Cluster failed during startup: {ev['error']}")

        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def _dispatch_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                ev = self.event_queue.get(timeout=0.2)
            except queue.Empty:
                # 巡检 worker 存活状态：有任意进程意外挂掉即触发整组回收
                dead = [p for p in self.procs if not p.is_alive() and p.exitcode not in (0, None)]
                if dead:
                    self._abort_cluster(f"Worker process died unexpectedly (exitcode={dead[0].exitcode})")
                    return
                continue

            etype = ev["type"]
            if etype == "STAGE_RELEASE":
                self.router.release_stage(ev["stage"], ev["rank"])
            elif etype == "TOKEN":
                with self._lock:
                    stream = self._streams.get(ev["req_id"])
                if stream is not None:
                    stream.push_token(ev)
            elif etype == "DONE":
                with self._lock:
                    stream = self._streams.get(ev["req_id"])
                if stream is not None:
                    stream.mark_done(ev)
            elif etype == "WORKER_ERROR":
                self._abort_cluster(f"rank={ev['rank']}({ev['role']}): {ev['error']}")
                return

    def _abort_cluster(self, reason: str) -> None:
        """异常时立刻唤醒所有在途请求和 FIFO 等待队列，并整组回收全部 worker"""
        with self._lock:
            if self._failed_reason is not None:
                return
            self._failed_reason = reason
            active_streams = list(self._streams.values())

        for s in active_streams:
            s.fail(reason)
        self.router.fail_all_waiters(RuntimeError(f"Cluster aborted: {reason}"))
        self._stop_flag.set()
        self._reap_workers(graceful=False)

    def _reap_workers(self, graceful: bool = True) -> None:
        if graceful:
            try:
                self.v_task_queue.put({"type": "STOP", "p_rank": 1})
            except Exception:
                pass
            for p in self.procs:
                p.join(timeout=5)
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=3)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)

    def stream_generate(
        self,
        req_id: str,
        prompt: str,
        images_bytes: List[bytes],
        max_new_tokens: int = 8,
        inject_crash: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        lease: RouteLease = self.router.acquire(req_id)
        stream = _RequestStream(req_id)
        with self._lock:
            if self._failed_reason is not None:
                raise RuntimeError(self._failed_reason)
            self._streams[req_id] = stream

        try:
            self.v_task_queue.put(
                {
                    "type": "TASK",
                    "req_id": req_id,
                    "p_rank": lease.p_rank,
                    "d_rank": lease.d_rank,
                    "prompt": prompt,
                    "images_bytes": images_bytes,
                    "max_new_tokens": max_new_tokens,
                    "inject_crash": inject_crash,
                }
            )
            while True:
                ev = stream.out_q.get(timeout=60)
                yield ev
                if ev["type"] in ("DONE", "ERROR"):
                    break
        finally:
            with self._lock:
                self._streams.pop(req_id, None)

    def shutdown(self) -> None:
        self._stop_flag.set()
        self._reap_workers(graceful=True)
