from typing import Any, Dict, Optional, Tuple
import torch
import torch.distributed as dist

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int64": torch.int64,
}


class VPDTransport:
    """控制面走 Gloo（元数据 + ACK），数据面可配置 gloo（单卡）或 nccl（多卡）。"""

    def __init__(self, data_backend: str = "gloo"):
        assert data_backend in ("gloo", "nccl"), f"Unsupported backend: {data_backend}"
        self.data_backend = data_backend
        self.ctrl_group = dist.group.WORLD
        self.data_group = dist.new_group(backend=data_backend)

    def send_control(self, msg: Dict[str, Any], dst_rank: int) -> None:
        dist.send_object_list([msg], dst=dst_rank, group=self.ctrl_group)

    def recv_control(self, src_rank: int) -> Dict[str, Any]:
        buf = [None]
        dist.recv_object_list(buf, src=src_rank, group=self.ctrl_group)
        return buf[0]

    def send_tensor_with_ack(
        self,
        tensor: torch.Tensor,
        meta: Dict[str, Any],
        dst_rank: int,
    ) -> Dict[str, Any]:
        dtype_name = str(tensor.dtype).replace("torch.", "")
        header = {
            **meta,
            "type": "TENSOR",
            "shape": list(tensor.shape),
            "dtype": dtype_name,
            "numel": int(tensor.numel()),
        }
        self.send_control(header, dst_rank=dst_rank)

        if self.data_backend == "gloo":
            send_buf = tensor.detach().to("cpu").contiguous()
        else:
            send_buf = tensor.detach().contiguous()
            torch.cuda.synchronize(send_buf.device)

        dist.send(send_buf, dst=dst_rank, group=self.data_group)

        ack = self.recv_control(src_rank=dst_rank)
        if not ack.get("ok", False):
            raise RuntimeError(f"Handoff NACK from rank {dst_rank}: {ack}")
        return ack

    def recv_tensor_with_ack(
        self,
        src_rank: int,
        target_device: str,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        header = self.recv_control(src_rank=src_rank)
        if header.get("type") == "STOP":
            return header, None

        shape = tuple(header["shape"])
        dtype = DTYPE_MAP[header["dtype"]]
        recv_dev = "cpu" if self.data_backend == "gloo" else target_device
        recv_buf = torch.empty(shape, dtype=dtype, device=recv_dev)

        dist.recv(recv_buf, src=src_rank, group=self.data_group)

        ok = (tuple(recv_buf.shape) == shape) and (int(recv_buf.numel()) == header["numel"])
        tensor = recv_buf.to(target_device, non_blocking=False) if recv_dev != target_device else recv_buf
        if target_device.startswith("cuda"):
            torch.cuda.synchronize(target_device)

        self.send_control(
            {"type": "ACK", "req_id": header.get("req_id"), "ok": ok},
            dst_rank=src_rank,
        )
        if not ok:
            raise RuntimeError(f"Tensor verification failed for req {header.get('req_id')}")
        return header, tensor
