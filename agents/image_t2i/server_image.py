"""Starlette server for the text-to-image task agent."""

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

from shared.config import HOST, IMAGE_T2I_PORT, IMAGE_T2I_PUBLIC_URL, TRACING_API_BASE
from shared.logger import setup_logging

from .image_task_executor import ImageTaskAgentExecutor

logger = setup_logging("image_t2i_server")


class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):  # type: ignore[override]
        start = time.perf_counter()
        response = await call_next(req)
        duration = (time.perf_counter() - start) * 1000
        client_host = req.client.host if req.client else "-"
        logger.info("[HTTP] %s %s %s -> %s (%.1f ms)", client_host, req.method, req.url.path, response.status_code, duration)
        return response


if __name__ == "__main__":
    skill = AgentSkill(
        id="t2i_basic",
        name="Text-to-Image (Task)",
        description="Generate one or more images from a prompt using DashScope wan2.5-t2i-preview.",
        tags=["image", "t2i", "task"],
        examples=['{"prompt":"可爱的猫咪宇航员","n":1,"size":"1024*1024"}'],
    )

    agent_card = AgentCard(
        name=os.getenv("IMAGE_T2I_AGENT_NAME", "vision_image_t2i"),
        description="DashScope wan2.5 T2I with Holos tracing and Task API.",
        url=IMAGE_T2I_PUBLIC_URL,
        version="1.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="vision-agents", organization="vision-agents", url="https://example.org/"),
    )

    tracer = PlantTracer(base_url=TRACING_API_BASE, agent_card=agent_card)

    handler = HolosRequestHandler(
        agent_executor=ImageTaskAgentExecutor(),
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    app = server.build()
    app.add_middleware(AccessLog)

    @app.get("/health")
    async def health(_):
        return JSONResponse({"status": "ok", "version": agent_card.version})

    uvicorn.run(app, host=HOST, port=IMAGE_T2I_PORT, log_level="info")
