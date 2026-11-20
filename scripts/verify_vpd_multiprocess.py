import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from lmm_serving.vpd.model_stages import run_vision_stage, run_prefill_stage, run_decode_stage
from lmm_serving.vpd.transport import VPDTransport

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
GOLDEN_IDS = [378, 2443, 335, 260, 2049, 553, 253, 5681]
MAX_NEW_TOKENS = 8


def worker_main(rank: int, world_size: int, data_backend: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    os.environ["HF_HUB_OFFLINE"] = "1"

    device = f"cuda:{rank}" if data_backend == "nccl" else "cuda:0"
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    transport = VPDTransport(data_backend=data_backend)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    eos_id = processor.tokenizer.eos_token_id
    dist.barrier()

    if rank == 0:  # V0: Vision Worker
        img1 = Image.new("RGB", (384, 384), color=(220, 40, 40))
        img2 = Image.new("RGB", (384, 384), color=(40, 120, 220))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Describe the colors of these two images briefly."},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[img1, img2], return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        for req_idx, dst_decode in enumerate([2, 3]):
            req_id = f"req-{req_idx}"
            t0 = time.perf_counter()
            vision_feats = run_vision_stage(model, inputs["pixel_values"], inputs["pixel_attention_mask"])
            meta = {
                "req_id": req_id,
                "dst_decode": dst_decode,
                "input_ids": inputs["input_ids"].cpu().tolist(),
                "attention_mask": inputs["attention_mask"].cpu().tolist(),
                "max_new_tokens": MAX_NEW_TOKENS,
            }
            ack = transport.send_tensor_with_ack(vision_feats, meta=meta, dst_rank=1)
            del vision_feats
            print(f"[V0 rank=0] {req_id} -> P0 ACK={ack['ok']} ({(time.perf_counter()-t0)*1000:.1f}ms)", flush=True)

        transport.send_control({"type": "STOP"}, dst_rank=1)

    elif rank == 1:  # P0: Prefill Worker
        while True:
            header, vision_feats = transport.recv_tensor_with_ack(src_rank=0, target_device=device)
            if header.get("type") == "STOP":
                transport.send_control({"type": "STOP"}, dst_rank=2)
                transport.send_control({"type": "STOP"}, dst_rank=3)
                break

            req_id = header["req_id"]
            dst_decode = header["dst_decode"]
            input_ids = torch.tensor(header["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(header["attention_mask"], dtype=torch.long, device=device)

            t0 = time.perf_counter()
            first_token, packed_kv = run_prefill_stage(model, input_ids, attention_mask, vision_feats)
            del vision_feats

            meta = {
                "req_id": req_id,
                "first_token": first_token.cpu().tolist(),
                "attention_mask": header["attention_mask"],
                "max_new_tokens": header["max_new_tokens"],
            }
            ack = transport.send_tensor_with_ack(packed_kv, meta=meta, dst_rank=dst_decode)
            del packed_kv
            print(
                f"[P0 rank=1] {req_id} -> D{dst_decode-2}(rank={dst_decode}) KV ACK={ack['ok']} "
                f"({(time.perf_counter()-t0)*1000:.1f}ms)",
                flush=True,
            )

    else:  # rank 2 (D0) & rank 3 (D1)
        while True:
            header, packed_kv = transport.recv_tensor_with_ack(src_rank=1, target_device=device)
            if header.get("type") == "STOP":
                break

            req_id = header["req_id"]
            first_token = torch.tensor(header["first_token"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(header["attention_mask"], dtype=torch.long, device=device)

            t0 = time.perf_counter()
            tokens = run_decode_stage(
                model,
                first_token,
                attention_mask,
                packed_kv,
                max_new_tokens=header["max_new_tokens"],
                eos_token_id=eos_id,
            )
            del packed_kv
            ids = tokens[0].tolist()
            text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            assert ids == GOLDEN_IDS, f"[D{rank-2}] Mismatch: {ids} != {GOLDEN_IDS}"
            print(f"[D{rank-2} rank={rank}] {req_id} PASS ({(time.perf_counter()-t0)*1000:.1f}ms) -> {text!r}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker_main, args=(4, "gloo"), nprocs=4, join=True)
