# ModServe-VPD: 多模态三阶段分离式推理服务

cd /root/lanyun-tmp/mini-sglang/ [Azure/ModServe](https://github.com/Azure/ModServe)（SoCC 2025）扩展，使用 PyTorch / Transformers 为 [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) 实现**视觉编码（V）+ Prefill（P）+ Decode（D）三阶段分离式推理后端**，支持可配置设备池、Gloo/NCCL 双通道交接、原子预留路由背压、SSE 流式输出与 BF16 精确对齐的视觉特征 LRU 缓存。

## 核心架构

```text
Client (HTTP / SSE)
   │
   ▼
Gateway & VPDRouter (FastAPI + 原子预留槽位 + FIFO 队列 + HTTP 429 背压 + Token 重排)
   │  Gloo 控制面 (请求任务)
   ▼
Vision Pool (1V) ──[NCCL/Gloo 视觉特征 + Gloo ACK]──► Prefill Pool (1P)
  · 内容寻址 SHA256 LRU (按字节预算淘汰)                · 注入视觉特征跑 Prefill
  · 仅保留 vision_model + connector (-62.7% 显存)      · 立即回传首 Token (压低 TTFT)
  · 请求形状隔离 + 非零占位补齐 (0.0 BF16 漂移)         · 打包 30 层连续 KV
                                                          │
                                       [NCCL/Gloo 打包 KV + Gloo ACK]
                                                          ▼
                                                 Decode Pool (2D: D0 / D1)
                                                  · 双副本并行增量解码
                                                  · 仅保留 text_model + lm_head (-37.3% 显存)
                                                  · 逐 Token 流式回传网关
```

## 核心特性

- **三阶段拆分与权重裁剪（`src/lmm_serving/vpd/model_stages.py`）**：将 V / P / D 拆为独立阶段并按需裁剪冗余子模块权重（`V` 显存 `496.8MB → 185.5MB`，`P/D` 显存 `496.8MB → 311.4MB`）；将 30 层 `(K, V)` 打包为单个连续张量 `[L, 2, B, H, S, D]`，消除逐层通信开销。
- **双通道通信与安全交接（`src/lmm_serving/vpd/transport.py`）**：控制面走 Gloo 传递元数据、张量形状与校验 ACK，数据面支持 `gloo`（单卡/共享卡调试）与 `nccl`（多卡 P2P 直传）一键切换；接收端校验并接管显存后发送端再释放引用。
- **原子预留路由与异常整组回收（`src/lmm_serving/vpd/scheduler.py`, `cluster.py`, `server.py`）**：入队时原子预留 `(V, P, D)` 槽位避免跨阶段死锁，V/P 在收到交接 ACK 后立即释放、D 在生成结束后释放，实现跨请求阶段重叠与双 Decode 副本并行；支持 FIFO 等待队列、接纳上限背压（HTTP 429）、Token 序号重排 SSE 输出，以及 worker 异常时即时唤醒在途请求并整组回收进程。
- **视觉特征 LRU 与 BF16 数值修复（`src/lmm_serving/vpd/vision_cache.py`）**：实现基于内容 SHA256 寻址与字节预算淘汰的视觉特征 LRU；定位并修复多图部分命中改变 encoder batch 形状导致的 BF16 漂移（`1,006,693 / 1,253,376` 元素漂移、最大误差 `1.0` → 按请求形状隔离键 + 非零占位补齐原满 batch 形状后误差归零至 `0.0`，与单模型逐 Token 100% 一致）。

## 实测结果（2×A100-PCIE-40GB, PIX）

cd /root/lanyun-tmp/mini-sglang`SmolVLM-256M-Instruct` BF16、双图（34 tiles）/ 8 输出 Token、客户端并发 1、90% 图片命中率混合复用负载。

| 指标 | 无缓存基线 | 启用视觉 LRU 缓存 | 收益 / 校验 |
| :--- | :--- | :--- | :--- |
| **TTFT P50（真 NCCL 通道）** | `104.5 ms` | **`73.1 ms`** | **降低 30.1%** |
| **30 层打包 KV 交接 + ACK P50** | `49.94 MB` | `6.31 ms` | 有效带宽 `7.73 GB/s` |
| **与单模型逐 Token 对齐** | 40 / 40 | 40 / 40 | **0 误差（BF16 漂移 0.0）** |
| **按阶段权重显存裁剪** | `496.8 MB / 进程` | `V: 185.5 MB` / `P,D: 311.4 MB` | **V -62.7% / P,D -37.3%** |

## 目录结构

```text
src/lmm_serving/vpd/
 model_stages.py   # SmolVLM 三阶段前向、30 层 KV 打包/解包、按阶段权重裁剪
 transport.py      # Gloo 控制面/ACK + 可插拔 Gloo/NCCL 数据面
 scheduler.py      # 可配置设备池、原子预留路由、FIFO 等待队列与背压
 cluster.py        # 多进程池管理、可配置 device_map、Token 重排与异常整组回收
 server.py         # FastAPI + SSE 流式服务入口 (/v1/generate, /health)
 vision_cache.py   # SHA256 字节预算 LRU、BF16 形状隔离与非零占位补齐

scripts/
 verify_smolvlm_vpd_local.py      # 单卡三阶段前向拆分与单模型逐 Token 对齐
 verify_vpd_multiprocess.py       # 1V+1P+2D 多进程 Gloo/NCCL 交接验证
 verify_vpd_service.py            # 并发重叠、双 D 并行、Token 重排、背压与异常回收验证
 verify_vision_cache_bf16.py      # LRU 字节淘汰 + Naive BF16 漂移复现与 0.0 修复验证
 bench_vpd_cache.py               # 1V+1P+2D 可配置 device_map 缓存压测
 bench_2gpu_nccl.py               # 双卡真 NCCL 分离部署压测
 bench_2gpu_nccl_gpu_preproc.py   # 双卡真 NCCL + GPU 34-tile 切图压测
```

## 快速开始

```bash
conda create -n modserve-vpd python=3.11 -y
conda activate modserve-vpd
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install "transformers>=4.49.0" accelerate pillow fastapi uvicorn sse-starlette pyyaml pytest requests pandas
pip install -e . --no-deps

# 1. 单卡三阶段逐 Token 对齐验证
python scripts/verify_smolvlm_vpd_local.py

# 2. 1V+1P+2D 多进程交接、调度背压与异常回收验证
HF_HUB_OFFLINE=1 python scripts/verify_vpd_service.py

# 3. 视觉 LRU 与 BF16 漂移修复验证
HF_HUB_OFFLINE=1 python scripts/verify_vision_cache_bf16.py

# 4. 双卡真 NCCL 分离部署与缓存收益压测（需 >= 2 GPU）
HF_HUB_OFFLINE=1 python scripts/bench_2gpu_nccl_gpu_preproc.py --num-requests 40 --hit-rate 0.90

# 5. 双卡/四卡 1V+1P+2D 全流程压测（四卡可切 --backend nccl）
HF_HUB_OFFLINE=1 python scripts/bench_vpd_cache.py --num-requests 40 --hit-rate 0.90 --backend gloo --device-map 0,0,1,1
```

## 致谢

 [Azure/ModServe](https://github.com/Azure/ModServe) 开源框架扩展开发，原论文发表于 ACM SoCC 2025（*ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving*）。
