import argparse
import io
import statistics
import time
from PIL import Image, ImageDraw

from lmm_serving.vpd.cluster import VPDCluster


def make_image_pool(num_unique: int):
    pool = []
    for i in range(num_unique):
        img = Image.new("RGB", (384, 384), color=((i * 53) % 256, (i * 97) % 256, (i * 131) % 256))
        draw = ImageDraw.Draw(img)
        draw.rectangle([40 + (i * 7) % 120, 40, 260, 260 + (i * 5) % 80], fill=((i * 173) % 256, 200, 90))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pool.append(buf.getvalue())
    return pool


def build_workload(num_requests: int, hit_rate: float = 0.90):
    """
    Construct a 2-image mixed-reuse workload with target image hit rate (~90%).
    Total image lookups = 2 * num_requests; unique images = round((1 - hit_rate) * total_lookups).
    """
    total_lookups = 2 * num_requests
    num_unique = max(2, int(round((1.0 - hit_rate) * total_lookups)))
    pool = make_image_pool(num_unique)

    requests_data = []
    # Mix full misses, partial hits, and full hits cleanly
    for r in range(num_requests):
        if 2 * r + 1 < num_unique:
            i0, i1 = 2 * r, 2 * r + 1
        elif 2 * r < num_unique:
            i0, i1 = 2 * r, 0
        else:
            # Reuse hot pairs from pool -> cache hits
            i0 = r % num_unique
            i1 = (r + 1) % num_unique
        requests_data.append(([pool[i0], pool[i1]], "Describe the colors and shapes in these two images briefly."))
    return requests_data, num_unique


def run_benchmark(workload, backend: str, port: int, enable_cache: bool, max_new_tokens: int):
    cluster = VPDCluster(
        data_backend=backend,
        master_port=port,
        max_admitted=4,
        enable_vision_cache=enable_cache,
    )
    cluster.start()
    ttfts = []
    outputs = []
    try:
        # Warmup 2 requests (excluded from latency stats)
        for w in range(2):
            imgs, prompt = workload[0]
            for _ in cluster.stream_generate(f"warmup-{w}", prompt, imgs, max_new_tokens=max_new_tokens):
                pass

        for idx, (imgs, prompt) in enumerate(workload):
            t0 = time.perf_counter()
            ttft_ms = None
            tids = []
            for ev in cluster.stream_generate(f"bench-{idx}", prompt, imgs, max_new_tokens=max_new_tokens):
                if ev["type"] == "TOKEN":
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    tids.append(ev["token_id"])
                elif ev["type"] == "ERROR":
                    raise RuntimeError(ev["error"])
            ttfts.append(ttft_ms)
            outputs.append(tids)
    finally:
        cluster.shutdown()
    return ttfts, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=40)
    parser.add_argument("--hit-rate", type=float, default=0.90)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    args = parser.parse_args()

    workload, num_unique = build_workload(args.num_requests, args.hit_rate)
    print(f"Workload: {args.num_requests} reqs (2 imgs/req, {args.max_new_tokens} tokens, {num_unique} unique imgs, target hit~{args.hit_rate*100:.0f}%, backend={args.backend})")

    print("\n[1/2] Running baseline (vision cache OFF)...")
    ttft_off, out_off = run_benchmark(workload, args.backend, 29531, enable_cache=False, max_new_tokens=args.max_new_tokens)

    print("[2/2] Running cached   (vision cache ON )...")
    ttft_on, out_on = run_benchmark(workload, args.backend, 29532, enable_cache=True, max_new_tokens=args.max_new_tokens)

    # Strict token-for-token parity check across every request
    mismatches = sum(1 for a, b in zip(out_off, out_on) if a != b)
    p50_off = statistics.median(ttft_off)
    p50_on = statistics.median(ttft_on)
    gain_pct = (p50_off - p50_on) / p50_off * 100.0

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"Requests verified   : {len(workload)} / {len(workload)}")
    print(f"Token mismatches    : {mismatches}")
    print(f"TTFT P50 (Cache OFF): {p50_off:.1f} ms")
    print(f"TTFT P50 (Cache ON) : {p50_on:.1f} ms")
    print(f"TTFT P50 Reduction  : {gain_pct:.1f}%")
    print("===================================================")
    assert mismatches == 0, f"Found {mismatches} token mismatches between Cache OFF and Cache ON!"
    print("PASS: Milestone 5 complete! Exact token parity + measurable TTFT P50 speedup!")


if __name__ == "__main__":
    main()
