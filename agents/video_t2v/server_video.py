"""Starlette server for the text-to-video task agent."""

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

from shared.config import HOST, TRACING_API_BASE, VIDEO_T2V_PORT, VIDEO_T2V_PUBLIC_URL
from shared.logger import setup_logging

from .video_task_executor import VideoTaskAgentExecutor

logger = setup_logging("video_t2v_server")


class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):  # type: ignore[override]
        start = time.perf_counter()
        response = await call_next(req)
        duration = (time.perf_counter() - start) * 1000
        client_host = req.client.host if req.client else "-"
        logger.info(
            "[HTTP] %s %s %s -> %s (%.1f ms)",
            client_host,
            req.method,
            req.url.path,
            response.status_code,
            duration,
        )
        return response


if __name__ == "__main__":
    skill_basic = AgentSkill(
        id="t2v_basic",
        name="Text-to-Video (Task)",
        description="Generate a short video from text; returns Task + Artifact with video URL and local path.",
        tags=["video", "t2v", "task"],
        examples=['{"prompt":"史诗可爱小猫将军在悬崖上","duration":8,"size":"832*480"}'],
    )
    agent_card = AgentCard(
        name=os.getenv("VIDEO_AGENT_NAME", "vision_video_t2v"),
        description="DashScope wan2.5 T2V with Holos tracing and Task API.",
        url=VIDEO_T2V_PUBLIC_URL,
        version="1.2.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill_basic],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="vision-agents", organization="vision-agents", url="https://example.org/"),
    )

    tracer = PlantTracer(base_url=TRACING_API_BASE, agent_card=agent_card)

    request_handler = HolosRequestHandler(
        agent_executor=VideoTaskAgentExecutor(),
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    app = server.build()
    app.add_middleware(AccessLog)

    @app.get("/health")
    async def health(_):
        return JSONResponse({"status": "ok", "version": agent_card.version})

    uvicorn.run(app, host=HOST, port=VIDEO_T2V_PORT, log_level="info")
