# server_image_edit.py
# -*- coding: utf-8 -*-
import os, time, logging, uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider

from holos_sdk import HolosRequestHandler, PlantTracer
from image_edit_task_executor import ImageEditTaskAgentExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(name)s:%(lineno)d [%(levelname)s] %(message)s")
logger = logging.getLogger("image_edit_server")

class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):
        t0 = time.perf_counter()
        resp = await call_next(req)
        logger.info(f"[HTTP] {req.client.host if req.client else '-'} {req.method} {req.url.path} -> {resp.status_code} ({(time.perf_counter()-t0)*1000:.1f} ms)")
        return resp

if __name__ == "__main__":
    PORT = int(os.getenv("IMAGE_EDIT_PORT", "10120"))
    PUBLIC_URL = os.getenv(
        "IMAGE_EDIT_PUBLIC_URL",
        "https://nat-notebook-inspire.sii.edu.cn/ws-6e6ca4ba-053a-4106-a019-0562c8f9364f/project-585056d0-0dd2-4222-86b6-97158106e509/user-7be56baa-199e-4e4c-a1b5-2b68ef52567b/vscode/9c91ff63-1562-44b9-bd55-bca85fddac53/6b7f2d83-99b4-41e5-90cb-d7f0643a1914/proxy/10120/"
    )

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
        name=os.getenv("IMAGE_EDIT_AGENT_NAME", "haoxiangzhaotestimageedit"),
        description="Image editing via DashScope qwen-image-edit-plus. Accepts local paths and URLs.",
        url=PUBLIC_URL,
        version="1.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(name="haoxiangzhao", organization="001-haoxiangzhao", url="https://example.com"),
    )

    tracer = PlantTracer(
        base_url=os.getenv(
            "TRACING_API_BASE",
            "https://nat-notebook-inspire.sii.edu.cn/ws-677a4cfa-ac67-494c-b481-e9147a3487a2/project-034ec99f-f57a-4c71-9f7e-1654d2c430c8/user-dc5518ff-a2c5-45ed-a36f-3931cb726d94/vscode/9b87a33b-9112-4cf2-90b9-de6b42e6072a/c8f23942-50b9-4630-ae9a-ce48e66b7281/proxy/8000/api/v1"
        ),
        agent_card=public_card
    )

    handler = HolosRequestHandler(
        agent_executor=ImageEditTaskAgentExecutor(),   # ✅ 不再传硬编码 key，走环境变量
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    server = A2AStarletteApplication(agent_card=public_card, http_handler=handler)
    app = server.build()
    app.add_middleware(AccessLog)

    @app.get("/health")
    async def health(_):
        return JSONResponse({"status":"ok","name":public_card.name,"version":public_card.version})

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
