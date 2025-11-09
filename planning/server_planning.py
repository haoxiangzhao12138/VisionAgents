"""Standalone server for the planning agent."""

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

from shared.config import HOST, PLANNING_PORT, PLANNING_PUBLIC_URL, TRACING_API_BASE
from shared.logger import setup_logging

from .planning_agent_executor import PlanningAgentExecutor

logger = setup_logging("planning_server")


class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):  # type: ignore[override]
        start = time.perf_counter()
        response = await call_next(req)
        duration = (time.perf_counter() - start) * 1000
        logger.info("[HTTP] %s %s -> %s (%.1f ms)", req.method, req.url.path, response.status_code, duration)
        return response


if __name__ == "__main__":
    skill = AgentSkill(
        id="visual_planning",
        name="Visual Planning (Holos Plan)",
        description="Decompose user intent into a Holos Plan, inject prompt_boost if needed, and forward to Router.",
        tags=["planning", "router", "holos", "plan"],
        examples=['"请先生成一张图，再把它变成视频"'],
    )
    agent_card = AgentCard(
        name=os.getenv("PLANNING_AGENT_NAME", "vision_planning"),
        description="Planning agent using Yunstorm GPT to create Holos Plan and forward to Router.",
        url=PLANNING_PUBLIC_URL,
        version="1.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="vision-agents", organization="vision-agents", url="https://example.org/"),
    )

    tracer = PlantTracer(base_url=TRACING_API_BASE, agent_card=agent_card)

    request_handler = HolosRequestHandler(
        agent_executor=PlanningAgentExecutor(),
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler).build()
    app.add_middleware(AccessLog)

    @app.route("/health")
    async def health(_):
        return JSONResponse({"status": "ok", "version": agent_card.version})

    uvicorn.run(app, host=HOST, port=PLANNING_PORT, log_level="info")
