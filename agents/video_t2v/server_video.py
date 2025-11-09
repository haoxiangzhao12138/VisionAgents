# server_video.py
import uvicorn, os, time, logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider
from holos_sdk import HolosRequestHandler, PlantTracer
from video_task_executor import VideoTaskAgentExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("video_server")

class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):
        t0 = time.perf_counter()
        resp = await call_next(req)
        logger.info(f"[HTTP] {req.client.host if req.client else '-'} {req.method} {req.url.path} -> {resp.status_code} ({(time.perf_counter()-t0)*1000:.1f} ms)")
        return resp

if __name__ == "__main__":
    PORT = int(os.getenv("VIDEO_PORT", "8889"))
    PUBLIC_URL = os.getenv(
        "VIDEO_PUBLIC_URL",
        "https://nat-notebook-inspire.sii.edu.cn/ws-6e6ca4ba-053a-4106-a019-0562c8f9364f/project-585056d0-0dd2-4222-86b6-97158106e509/user-7be56baa-199e-4e4c-a1b5-2b68ef52567b/vscode/9c91ff63-1562-44b9-bd55-bca85fddac53/6b7f2d83-99b4-41e5-90cb-d7f0643a1914/proxy/8889/"
    )

    skill_basic = AgentSkill(
        id="t2v_basic",
        name="Text-to-Video (Task)",
        description="Generate a short video from text; returns Task + Artifact with video URL and local path.",
        tags=["video","t2v","task"],
        examples=['{"prompt":"史诗可爱小猫将军在悬崖上","duration":8,"size":"832*480"}'],
    )
    agent_card = AgentCard(
        name=os.getenv("VIDEO_AGENT_NAME", "hxztest_video3"),
        description="DashScope wan2.5 T2V with Holos tracing and Task API.",
        url=PUBLIC_URL,
        version="1.2.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill_basic],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="haoxiangzhao", organization="001-haoxiangzhao", url="https://haoxiangzhao12138.github.io/"),
    )

    tracer = PlantTracer(
        base_url=os.getenv(
            "TRACING_API_BASE",
            "https://nat-notebook-inspire.sii.edu.cn/ws-677a4cfa-ac67-494c-b481-e9147a3487a2/project-034ec99f-f57a-4c71-9f7e-1654d2c430c8/user-dc5518ff-a2c5-45ed-a36f-3931cb726d94/vscode/9b87a33b-9112-4cf2-90b9-de6b42e6072a/c8f23942-50b9-4630-ae9a-ce48e66b7281/proxy/8000/api/v1"
        ),
        agent_card=agent_card
    )

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
        return JSONResponse({"status":"ok", "version":agent_card.version})

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
