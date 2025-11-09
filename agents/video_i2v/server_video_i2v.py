"""Starlette server for the image-to-video task agent."""

from __future__ import annotations

import os
import time

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentProvider, AgentSkill
from holos_sdk import HolosRequestHandler, PlantTracer

from shared.config import HOST, TRACING_API_BASE, VIDEO_I2V_PORT, VIDEO_I2V_PUBLIC_URL
from shared.logger import setup_logging

from .video_i2v_task_executor import VideoI2VTaskAgentExecutor

logger = setup_logging("video_i2v_server")


class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):  # type: ignore[override]
        start = time.perf_counter()
        response = await call_next(req)
        duration = (time.perf_counter() - start) * 1000
        client_host = req.client.host if req.client else "-"
        logger.info("[HTTP] %s %s %s -> %s (%.1f ms)", client_host, req.method, req.url.path, response.status_code, duration)
        return response


if __name__ == "__main__":
    skill_i2v = AgentSkill(
        id="i2v_basic",
        name="Image-to-Video (Task)",
        description="Generate a short video from 1 image using DashScope wan2.5-i2v-preview; emits Task + Artifacts.",
        tags=["video", "i2v", "task"],
        examples=[
            '{"op":"i2v","img_url":"file:///data/cat.png","prompt":"动感都市涂鸦风","duration":8,"resolution":"480P"}',
            '{"op":"i2v","img_url":{"$ref":"t2i1.results.0.url"},"prompt":{"$ref":"boost1.enhanced_prompt"}}',
        ],
    )

    agent_card = AgentCard(
        name=os.getenv("I2V_AGENT_NAME", "vision_video_i2v"),
        description="DashScope wan2.5 I2V with Holos tracing and Task API.",
        url=VIDEO_I2V_PUBLIC_URL,
        version="1.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill_i2v],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="vision-agents", organization="vision-agents", url="https://example.org/"),
    )

    tracer = PlantTracer(base_url=TRACING_API_BASE, agent_card=agent_card)

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

    uvicorn.run(app, host=HOST, port=VIDEO_I2V_PORT, log_level="info")
