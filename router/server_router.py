# -*- coding: utf-8 -*-
import os, time, logging, uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider

from holos_sdk import HolosRequestHandler, PlantTracer
from .router_agent_executor import RouterAgentExecutor
from shared.config import TRACING_API_BASE, ROUTER_PORT, ROUTER_PUBLIC_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("router_server")

class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):
        t0 = time.perf_counter()
        resp = await call_next(req)
        logger.info(f"[HTTP] {req.client.host if req.client else '-'} {req.method} {req.url.path} -> {resp.status_code} ({(time.perf_counter()-t0)*1000:.1f} ms)")
        return resp

if __name__ == "__main__":
    skill = AgentSkill(
        id="visual_router",
        name="Visual Router (Holos Plan)",
        description="Accepts a Holos Plan and dispatches subplans to task agents with Assignments and tracing.",
        tags=["router","plan","holos"],
        examples=["<Plan object>"],
    )
    agent_card = AgentCard(
        name=os.getenv("ROUTER_AGENT_NAME","hxz_router"),
        description="Routing agent that turns Plan nodes into task calls and emits Assignments.",
        url=ROUTER_PUBLIC_URL,
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
        agent_executor=RouterAgentExecutor(),
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    app = server.build()
    app.add_middleware(AccessLog)

    @app.get("/health")
    async def health(_):
        return JSONResponse({"status":"ok","version":agent_card.version})

    uvicorn.run(app, host="0.0.0.0", port=ROUTER_PORT, log_level="info")