"""Starlette server for the image edit task agent."""

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

from shared.config import HOST, IMAGE_EDIT_PORT, IMAGE_EDIT_PUBLIC_URL, TRACING_API_BASE
from shared.logger import setup_logging

from .image_edit_task_executor import ImageEditTaskAgentExecutor

logger = setup_logging("image_edit_server")


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
        id="image_edit",
        name="Image Edit (qwen-image-edit-plus)",
        description="Edit 1-3 images with an instruction. Supports local file paths or URLs. Returns n edited images.",
        tags=["image", "edit", "multimodal"],
        examples=[
            '{"instruction":"图1的人穿图2的黑裙子按图3姿势","images":["/data/img1.png","/data/img2.jpg","/data/pose.png"],"n":2}',
            '把第一张图调成日系胶片风格（本地路径：/data/img1.png）',
        ],
    )

    public_card = AgentCard(
        name=os.getenv("IMAGE_EDIT_AGENT_NAME", "vision_image_edit"),
        description="Image editing via DashScope qwen-image-edit-plus. Accepts local paths and URLs.",
        url=IMAGE_EDIT_PUBLIC_URL,
        version="1.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="vision-agents", organization="vision-agents", url="https://example.org/"),
    )

    tracer = PlantTracer(base_url=TRACING_API_BASE, agent_card=public_card)

    handler = HolosRequestHandler(
        agent_executor=ImageEditTaskAgentExecutor(),
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    server = A2AStarletteApplication(agent_card=public_card, http_handler=handler)
    app = server.build()
    app.add_middleware(AccessLog)

    @app.get("/health")
    async def health(_):
        return JSONResponse({"status": "ok", "name": public_card.name, "version": public_card.version})

    uvicorn.run(app, host=HOST, port=IMAGE_EDIT_PORT, log_level="info")
