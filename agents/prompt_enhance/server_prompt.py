# server_prompt.py
"""Starlette server for the prompt enhancement agent."""

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

from shared.config import HOST, PROMPT_PORT, PROMPT_PUBLIC_URL, TRACING_API_BASE
from shared.logger import setup_logging

from .prompt_enhance_executor import PromptEnhanceAgentExecutor

logger = setup_logging("prompt_enhance_server")


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
        id="prompt_enhance",
        name="Prompt Enhancement for Visual Tasks",
        description="Enhance user prompts before image/video tasks (T2I, T2V, I2V, Image Edit).",
        tags=["prompt", "enhance", "visual"],
        examples=[
            "把穿红裙子的女生拍成赛博朋克风格的照片，夜晚的雨巷，霓虹反射",
            '{"task_type":"text_to_video","prompt":"一只小狗在草地上奔跑"}',
        ],
    )

    public_card = AgentCard(
        name=os.getenv("PROMPT_AGENT_NAME", "vision_prompt_enhance"),
        description="Enhance prompts before image/video generation.",
        url=PROMPT_PUBLIC_URL,
        version="1.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="vision-agents", organization="vision-agents", url="https://example.org/"),
    )

    tracer = PlantTracer(base_url=TRACING_API_BASE, agent_card=public_card)

    request_handler = HolosRequestHandler(
        agent_executor=PromptEnhanceAgentExecutor(),
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    app = A2AStarletteApplication(agent_card=public_card, http_handler=request_handler).build()
    app.add_middleware(AccessLog)

    @app.route("/health")
    async def health(_):
        return JSONResponse({"status": "ok", "version": public_card.version})

    uvicorn.run(app, host=HOST, port=PROMPT_PORT, log_level="info")
