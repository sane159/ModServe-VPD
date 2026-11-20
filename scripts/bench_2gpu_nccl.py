import argparse
import io
import os
import statistics
import time
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmm_serving.vpd.model_stages import (
    prune_model_for_stage,
    run_decode_stage,
    run_prefill_stage,
)
from lmm_serving.vpd.transport import VPDTransport
from lmm_serving.vpd.vision_cache import (
    VisionFeatureLRU,
    encode_images_with_cache,
    try_full_hit_fast_path,
)

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def build_workload(num_requests: int, hit_rate: float = 0.90):
    total_lookups = 2 * num_requests
    num_unique = max(2, int(round((1.0 - hit_rate) * total_lookups)))
    pool = []
    for i in range(num_unique):
        img = Image.new("RGB", (384, 384), color=((i * 53) % 256, (i * 97) % 256, (i * 131) % 256))
        draw = ImageDraw.Draw(img)
        draw.rectangle([40 + (i * 7) % 120, 40, 260, 260 + (i * 5) % 80], fill=((i * 173) % 256, 200, 90))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pool.append(buf.getvalue())

    reqs = []
    for r in range(num_requests):
        if 2 * r + 1 < num_unique:
            i0, i1 = 2 * r, 2 * r + 1
        elif 2 * r < num_unique:
            i0, i1 = 2 * r, 0
        else:
            i0, i1 = r % num_unique, (r + 1) % num_unique
        reqs.append(([pool[i0], pool[i1]], "Describe the colors and shapes in these two images briefly."))
    return reqs


def worker(rank: int, num_requests: int, hit_rate: float, max_new_tokens: int, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["HF_HUB_OFFLINE"] = "1"

    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    dist.init_process_group(backend="gloo", rank=rank, world_size=2)
    transport = VPDTransport(data_backend="nccl")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    eos_id = processor.tokenizer.eos_token_id

    if rank == 1:
        b_mb, a_mb = prune_model_for_stage(model, "D")
        print(f"[Rank 1 / GPU 1 Decode] weights pruned: {b_mb:.1f}MB -> {a_mb:.1f}MB", flush=True)
    else:
        mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[Rank 0 / GPU 0 Vision+Prefill] weights loaded: {mem_mb:.1f}MB", flush=True)

    workload = build_workload(num_requests, hit_rate)
    dist.barrier()

    def run_pass(enable_cache: bool):
        v_cache = VisionFeatureLRU(max_bytes=256 * 1024 * 1024) if enable_cache else None
        ttfts = []
        kv_xfer_ms = []
        kv_bytes = 0
        outputs = []

        stream = [workload[0], workload[0]] + workload
        for idx, (imgs_bytes, prompt) in enumerate(stream):
            is_warmup = idx < 2
            if rank == 0:
                t0 = time.perf_counter()
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
                prompt_text = processor.apply_chat_template(msgs, add_generation_prompt=True)

                fast = try_full_hit_fast_path(processor, prompt_text, imgs_bytes, v_cache)
                if fast is not None:
                    input_ids_cpu, attn_mask_cpu, vision_feats = fast
                    input_ids = input_ids_cpu.to(device)
                    attention_mask = attn_mask_cpu.to(device)
                else:
                    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in imgs_bytes]
                    inputs = processor(text=prompt_text, images=images, return_tensors="pt").to(device)
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                    vision_feats = encode_images_with_cache(
                        model, processor, images, imgs_bytes,
                        inputs["pixel_values"], inputs["pixel_attention_mask"],
                        v_cache, isolate_by_req_shape=True, pad_to_orig_batch=True,
                    )
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                first_token, packed_kv = run_prefill_stage(model, input_ids, attention_mask, vision_feats)
                torch.cuda.synchronize(device)
                ttft = (time.perf_counter() - t0) * 1000.0
                del vision_feats

                kv_bytes = packed_kv.element_size() * packed_kv.nelement()
                t_xfer = time.perf_counter()
                meta = {
                    "req_id": f"r-{idx}",
                    "first_token": first_token.cpu().tolist(),
                    "attention_mask": attention_mask.cpu().tolist(),
                    "max_new_tokens": max_new_tokens,
                }
                transport.send_tensor_with_ack(packed_kv, meta=meta, dst_rank=1)
                xfer_ms = (time.perf_counter() - t_xfer) * 1000.0
                del packed_kv

                done_msg = transport.recv_control(src_rank=1)
                if not is_warmup:
                    ttfts.append(ttft)
                    kv_xfer_ms.append(xfer_ms)
                    outputs.append(done_msg["token_ids"])
            else:
                header, packed_kv = transport.recv_tensor_with_ack(src_rank=0, target_device=device)
                first_token = torch.tensor(header["first_token"], dtype=torch.long, device=device)
                attention_mask = torch.tensor(header["attention_mask"], dtype=torch.long, device=device)
                tokens = run_decode_stage(model, first_token, attention_mask, packed_kv, header["max_new_tokens"], eos_id)
                del packed_kv
                transport.send_control({"token_ids": tokens[0].tolist()}, dst_rank=0)

        return ttfts, kv_xfer_ms, kv_bytes, outputs

    ttft_off, xfer_off, kv_bytes, out_off = run_pass(enable_cache=False)
    dist.barrier()
    ttft_on, xfer_on, _, out_on = run_pass(enable_cache=True)
    dist.barrier()

    if rank == 0:
        mismatches = sum(1 for a, b in zip(out_off, out_on) if a != b)
        p50_off = statistics.median(ttft_off)
        p50_on = statistics.median(ttft_on)
        p50_xfer = statistics.median(xfer_on)
        bw_gbps = (kv_bytes / (1024 ** 3)) / (p50_xfer / 1000.0)
        gain = (p50_off - p50_on) / p50_off * 100.0
        print("\n=========== 2x A100 PCIe REAL NCCL SUMMARY ===========")
        print(f"Requests verified          : {num_requests} / {num_requests}")
        print(f"Token mismatches           : {mismatches}")
        print(f"Packed KV payload          : {kv_bytes / (1024**2):.2f} MB (30 layers BF16)")
        print(f"NCCL KV Handoff+ACK P50    : {p50_xfer:.2f} ms ({bw_gbps:.2f} GB/s effective)")
        print(f"TTFT P50 (Cache OFF)       : {p50_off:.1f} ms")
        print(f"TTFT P50 (Cache ON)        : {p50_on:.1f} ms")
        print(f"TTFT P50 Reduction         : {gain:.1f}%")
        print("======================================================")
        assert mismatches == 0, f"Found {mismatches} token mismatches!"
        print("PASS: Dual-A100 PCIe real NCCL disaggregation verified!")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=40)
    parser.add_argument("--hit-rate", type=float, default=0.90)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    mp.spawn(worker, args=(args.num_requests, args.hit_rate, args.max_new_tokens, 29541), nprocs=2, join=True)


if __name__ == "__main__":
    main()
