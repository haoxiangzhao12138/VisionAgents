"""Entry-point for the router service."""

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

from shared.config import HOST, ROUTER_PORT, ROUTER_PUBLIC_URL, TRACING_API_BASE
from shared.logger import setup_logging

from .router_agent_executor import RouterAgentExecutor

logger = setup_logging("router_server")


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
        id="visual_router",
        name="Visual Router (Holos Plan)",
        description="Accepts a Holos Plan and dispatches subplans to task agents with Assignments and tracing.",
        tags=["router", "plan", "holos"],
        examples=["<Plan object>"],
    )
    agent_card = AgentCard(
        name=os.getenv("ROUTER_AGENT_NAME", "vision_router"),
        description="Routing agent that turns Plan nodes into task calls and emits Assignments.",
        url=ROUTER_PUBLIC_URL,
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
        agent_executor=RouterAgentExecutor(),
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    app = server.build()
    app.add_middleware(AccessLog)

    @app.get("/health")
    async def health(_):
        return JSONResponse({"status": "ok", "version": agent_card.version})

    uvicorn.run(app, host=HOST, port=ROUTER_PORT, log_level="info")