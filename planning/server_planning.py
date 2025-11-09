# planning/server_planning.py
# -*- coding: utf-8 -*-
import os, time, logging, uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider

from holos_sdk import HolosRequestHandler, PlantTracer
from planning_agent_executor import PlanningAgentExecutor
from shared.logger import setup_logging
from shared.config import TRACING_API_BASE

logger = setup_logging("planning_server")

class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):
        t0 = time.perf_counter()
        resp = await call_next(req)
        logger.info("[HTTP] %s %s -> %s (%.1f ms)",
            req.method, req.url.path, resp.status_code, (time.perf_counter()-t0)*1000)
        return resp

if __name__ == "__main__":
    PORT = int(os.getenv("PLANNING_PORT", "11111"))
    PUBLIC_URL = os.getenv("PLANNING_PUBLIC_URL", f"http://localhost:{PORT}/")

    skill = AgentSkill(
        id="visual_planning",
        name="Visual Planning (Holos Plan)",
        description="Decompose user intent into a Holos Plan, inject prompt_boost if needed, and forward to Router.",
        tags=["planning","router","holos","plan"],
        examples=['"请先生成一张图，再把它变成视频"'],
    )
    agent_card = AgentCard(
        name=os.getenv("PLANNING_AGENT_NAME","hxz_planning"),
        description="Planning agent using Yunstorm GPT to create Holos Plan and forward to Router.",
        url=PUBLIC_URL,
        version="1.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="haoxiangzhao", organization="001-haoxiangzhao", url="https://example.org/"),
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
        return JSONResponse({"status":"ok","version":agent_card.version})

    uvicorn.run(app, host=os.getenv("HOST","0.0.0.0"), port=PORT, log_level="info")
