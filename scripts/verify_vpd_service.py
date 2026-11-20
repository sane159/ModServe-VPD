import concurrent.futures
import io
import time
from PIL import Image

from lmm_serving.vpd.cluster import VPDCluster
from lmm_serving.vpd.scheduler import BackpressureError

GOLDEN_IDS = [378, 2443, 335, 260, 2049, 553, 253, 5681]


def make_test_images():
    out = []
    for color in [(220, 40, 40), (40, 120, 220)]:
        buf = io.BytesIO()
        Image.new("RGB", (384, 384), color=color).save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def run_one(cluster: VPDCluster, req_id: str, imgs, inject_crash: bool = False):
    t0 = time.perf_counter()
    ttft = None
    tokens = []
    d_rank = None
    for ev in cluster.stream_generate(
        req_id=req_id,
        prompt="Describe the colors of these two images briefly.",
        images_bytes=imgs,
        max_new_tokens=8,
        inject_crash=inject_crash,
    ):
        if ev["type"] == "TOKEN":
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            tokens.append((ev["seq_idx"], ev["token_id"]))
        elif ev["type"] == "DONE":
            d_rank = ev["d_rank"]
        elif ev["type"] == "ERROR":
            return {"req_id": req_id, "error": ev["error"]}
    return {
        "req_id": req_id,
        "seq_ok": [s for s, _ in tokens] == list(range(len(tokens))),
        "token_ids": [t for _, t in tokens],
        "d_rank": d_rank,
        "ttft_ms": ttft,
        "total_ms": (time.perf_counter() - t0) * 1000,
    }


def main():
    imgs = make_test_images()
    print("=== 1. 启动 1V+1P+2D 集群 (max_admitted=3) ===")
    cluster = VPDCluster(data_backend="gloo", master_port=29521, max_admitted=3)
    cluster.start()

    print("=== 2. 并发发送 4 个请求：验证阶段重叠、双 Decode 并行、Token 重排与背压 ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run_one, cluster, f"req-{i}", imgs) for i in range(4)]
        ok_results = []
        backpressured = 0
        for f in futs:
            try:
                res = f.result()
                ok_results.append(res)
                print(
                    f"  [OK] {res['req_id']} on D{res['d_rank']-2}(rank={res['d_rank']}) "
                    f"seq_ok={res['seq_ok']} TTFT={res['ttft_ms']:.1f}ms Total={res['total_ms']:.1f}ms"
                )
                assert res["seq_ok"], f"Token out of order: {res}"
                assert res["token_ids"] == GOLDEN_IDS, f"Mismatch: {res['token_ids']}"
            except BackpressureError as e:
                backpressured += 1
                print(f"  [429 Backpressure] {e}")

    used_decode_ranks = {r["d_rank"] for r in ok_results}
    print(f"  使用到的 Decode 副本 ranks: {sorted(used_decode_ranks)}, 背压拦截数: {backpressured}")
    assert len(used_decode_ranks) == 2, "Expected both D0(2) and D1(3) to be utilized!"
    assert backpressured >= 1, "Expected at least 1 request to hit max_admitted=3 backpressure!"

    print("=== 3. 注入异常：验证在途请求被唤醒 + 整组 worker 干净回收 ===")
    err_res = run_one(cluster, "req-crash", imgs, inject_crash=True)
    print(f"  [Woke up with error] {err_res}")
    assert "error" in err_res, "In-flight request was not woken up on crash!"
    time.sleep(0.5)
    alive = [p.pid for p in cluster.procs if p.is_alive()]
    print(f"  回收后存活 worker 数: {len(alive)} (期望 0)")
    assert len(alive) == 0, f"Workers not reaped: {alive}"
    print("PASS: 里程碑 3 全部验收通过（原子预留/双D并行/Token重排/背压/异常整组回收）！")


if __name__ == "__main__":
    main()
