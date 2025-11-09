# server_video_i2v.py
# -*- coding: utf-8 -*-
import os, time, logging, uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider
from holos_sdk import HolosRequestHandler, PlantTracer
from video_i2v_task_executor import VideoI2VTaskAgentExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("video_i2v_server")

class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):
        t0 = time.perf_counter()
        resp = await call_next(req)
        logger.info(f"[HTTP] {req.client.host if req.client else '-'} {req.method} {req.url.path} -> {resp.status_code} ({(time.perf_counter()-t0)*1000:.1f} ms)")
        return resp

if __name__ == "__main__":
    PORT = int(os.getenv("I2V_PORT", "10121"))
    PUBLIC_URL = os.getenv("I2V_PUBLIC_URL", f"http://localhost:{PORT}/")

    skill_i2v = AgentSkill(
        id="i2v_basic",
        name="Image-to-Video (Task)",
        description="Generate a short video from 1 image using DashScope wan2.5-i2v-preview; emits Task + Artifacts.",
        tags=["video", "i2v", "task"],
        examples=[
            '{"op":"i2v","img_url":"file:///data/cat.png","prompt":"动感都市涂鸦风","duration":8,"resolution":"480P"}',
            # 或结合 Router $ref： {"op":"i2v","img_url":{"$ref":"t2i1.results.0.path"},"prompt":{"$ref":"boost1.enhanced_prompt"}}
        ],
    )

    agent_card = AgentCard(
        name=os.getenv("I2V_AGENT_NAME", "hxz_video_i2v"),
        description="DashScope wan2.5 I2V with Holos tracing and Task API.",
        url=PUBLIC_URL,
        version="1.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill_i2v],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="haoxiangzhao", organization="001-haoxiangzhao", url="https://haoxiangzhao12138.github.io/"),
    )

    tracer = PlantTracer(
        base_url=os.getenv("TRACING_API_BASE", "http://localhost:8000/api/v1"),
        agent_card=agent_card
    )

    handler = HolosRequestHandler(
        agent_executor=VideoI2VTaskAgentExecutor(api_key=os.getenv("DASHSCOPE_API_KEY", "")),
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    app = server.build()
    app.add_middleware(AccessLog)

    @app.get("/health")
    async def health(_):
        return JSONResponse({"status": "ok", "version": agent_card.version})

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=PORT, log_level="info")
