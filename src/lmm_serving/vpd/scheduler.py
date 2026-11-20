from collections import deque
from dataclasses import dataclass
import threading
from typing import Deque, Dict, List, Optional, Tuple


class BackpressureError(RuntimeError):
    """超过系统接纳上限时抛出，网关映射为 HTTP 429"""


@dataclass
class RouteLease:
    req_id: str
    v_rank: int
    p_rank: int
    d_rank: int


@dataclass
class _Waiter:
    req_id: str
    event: threading.Event
    lease: Optional[RouteLease] = None
    error: Optional[Exception] = None


class VPDRouter:
    """
    可配置设备池路由器：
    - 原子预留 (V, P, D) 槽位，避免跨阶段死锁
    - V/P 在交接 ACK 后释放，D 在生成结束后释放
    - 结合严格 FIFO 等待队列与最大接纳上限实现背压
    """

    def __init__(
        self,
        vision_ranks: List[int],
        prefill_ranks: List[int],
        decode_ranks: List[int],
        v_slots: int = 1,
        p_slots: int = 1,
        d_slots: int = 1,
        max_admitted: int = 8,
    ):
        self.v_cap = {r: v_slots for r in vision_ranks}
        self.p_cap = {r: p_slots for r in prefill_ranks}
        self.d_cap = {r: d_slots for r in decode_ranks}

        self.v_used = {r: 0 for r in vision_ranks}
        self.p_used = {r: 0 for r in prefill_ranks}
        self.d_used = {r: 0 for r in decode_ranks}

        self.max_admitted = max_admitted
        self._in_system = 0
        self._waiters: Deque[_Waiter] = deque()
        self._closed_error: Optional[Exception] = None
        self._lock = threading.Lock()

    def _pick_least_loaded(self, used: Dict[int, int], cap: Dict[int, int]) -> Optional[int]:
        candidates = [r for r, u in used.items() if u < cap[r]]
        if not candidates:
            return None
        return min(candidates, key=lambda r: (used[r], r))

    def _try_atomic_reserve_locked(self, req_id: str) -> Optional[RouteLease]:
        v = self._pick_least_loaded(self.v_used, self.v_cap)
        p = self._pick_least_loaded(self.p_used, self.p_cap)
        d = self._pick_least_loaded(self.d_used, self.d_cap)
        if v is None or p is None or d is None:
            return None
        self.v_used[v] += 1
        self.p_used[p] += 1
        self.d_used[d] += 1
        return RouteLease(req_id=req_id, v_rank=v, p_rank=p, d_rank=d)

    def acquire(self, req_id: str, timeout: float = 30.0) -> RouteLease:
        with self._lock:
            if self._closed_error is not None:
                raise self._closed_error
            if self._in_system >= self.max_admitted:
                raise BackpressureError(
                    f"Admission limit reached ({self._in_system}/{self.max_admitted})"
                )
            self._in_system += 1

            # 只有队列为空时才允许直接预留，确保严格 FIFO 公平性
            if not self._waiters:
                lease = self._try_atomic_reserve_locked(req_id)
                if lease is not None:
                    return lease

            waiter = _Waiter(req_id=req_id, event=threading.Event())
            self._waiters.append(waiter)

        if not waiter.event.wait(timeout=timeout):
            with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._in_system -= 1
            raise TimeoutError(f"Request {req_id} timed out waiting in FIFO queue")

        if waiter.error is not None:
            raise waiter.error
        assert waiter.lease is not None
        return waiter.lease

    def release_stage(self, stage: str, rank: int) -> None:
        with self._lock:
            if stage == "V":
                self.v_used[rank] = max(0, self.v_used[rank] - 1)
            elif stage == "P":
                self.p_used[rank] = max(0, self.p_used[rank] - 1)
            elif stage == "D":
                self.d_used[rank] = max(0, self.d_used[rank] - 1)
                self._in_system = max(0, self._in_system - 1)
            else:
                raise ValueError(f"Unknown stage: {stage}")
            self._wake_fifo_locked()

    def _wake_fifo_locked(self) -> None:
        while self._waiters:
            head = self._waiters[0]
            lease = self._try_atomic_reserve_locked(head.req_id)
            if lease is None:
                break
            self._waiters.popleft()
            head.lease = lease
            head.event.set()

    def fail_all_waiters(self, error: Exception) -> None:
        """集群异常时立刻唤醒所有 FIFO 等待请求，避免客户端永久挂死"""
        with self._lock:
            self._closed_error = error
            while self._waiters:
                w = self._waiters.popleft()
                w.error = error
                w.event.set()
            self._in_system = 0

    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "in_system": self._in_system,
                "max_admitted": self.max_admitted,
                "fifo_waiting": len(self._waiters),
                "v_used": dict(self.v_used),
                "p_used": dict(self.p_used),
                "d_used": dict(self.d_used),
            }
