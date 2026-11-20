import base64
import json
import uuid
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from lmm_serving.vpd.cluster import VPDCluster
from lmm_serving.vpd.scheduler import BackpressureError


class GenerateRequest(BaseModel):
    prompt: str
    images_b64: List[str]
    max_new_tokens: int = 8


def create_app(data_backend: str = "gloo", max_admitted: int = 4) -> FastAPI:
    app = FastAPI(title="ModServe VPD 3-Stage Server")
    cluster = VPDCluster(data_backend=data_backend, max_admitted=max_admitted)

    @app.on_event("startup")
    def _startup():
        cluster.start()

    @app.on_event("shutdown")
    def _shutdown():
        cluster.shutdown()

    @app.get("/health")
    def health():
        return {"status": "ok", "router": cluster.router.stats()}

    @app.post("/v1/generate")
    def generate(req: GenerateRequest):
        req_id = f"req-{uuid.uuid4().hex[:8]}"
        images_bytes = [base64.b64decode(b) for b in req.images_b64]

        def event_gen():
            try:
                for ev in cluster.stream_generate(
                    req_id=req_id,
                    prompt=req.prompt,
                    images_bytes=images_bytes,
                    max_new_tokens=req.max_new_tokens,
                ):
                    yield {"event": ev["type"].lower(), "data": json.dumps(ev, ensure_ascii=False)}
            except BackpressureError as e:
                yield {"event": "error", "data": json.dumps({"error": str(e), "code": 429})}

        # 提前检查背压，超过接纳上限直接返回 HTTP 429
        stats = cluster.router.stats()
        if stats["in_system"] >= stats["max_admitted"]:
            raise HTTPException(status_code=429, detail="Too many requests (backpressure)")

        return EventSourceResponse(event_gen())

    return app
