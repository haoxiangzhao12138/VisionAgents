# server_prompt.py
# -*- coding: utf-8 -*-
import os
import time
import logging
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentProvider

from holos_sdk import HolosRequestHandler, PlantTracer
from prompt_enhance_executor import PromptEnhanceAgentExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_server")


class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):
        t0 = time.perf_counter()
        resp = await call_next(req)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("[HTTP] %s %s -> %s (%.1f ms)",
                    req.method, req.url.path, resp.status_code, ms)
        return resp


if __name__ == "__main__":
    # === AgentCard（对外能力声明）===
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
        name="haoxiangzhao_prompt_enhance",
        description="Enhance prompts before image/video generation.",
        url="https://nat-notebook-inspire.sii.edu.cn/ws-6e6ca4ba-053a-4106-a019-0562c8f9364f/project-585056d0-0dd2-4222-86b6-97158106e509/user-7be56baa-199e-4e4c-a1b5-2b68ef52567b/vscode/9c91ff63-1562-44b9-bd55-bca85fddac53/6b7f2d83-99b4-41e5-90cb-d7f0643a1914/proxy/10130/",
        version="1.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
        provider=AgentProvider(
            name="haoxiangzhao",
            organization="001-haoxiangzhao",
            url="https://haoxiangzhao12138.github.io/",
        ),
    )

    # === Holos Tracer（很关键！）===
    # 优先读环境变量 HOLOS_TRACER_BASE_URL；没有就用老师提供的固定地址
    TRACER_BASE_URL = os.getenv(
        "HOLOS_TRACER_BASE_URL",
        "https://nat-notebook-inspire.sii.edu.cn/ws-677a4cfa-ac67-494c-b481-e9147a3487a2/"
        "project-034ec99f-f57a-4c71-9f7e-1654d2c430c8/user-dc5518ff-a2c5-45ed-a36f-3931cb726d94/"
        "vscode/9b87a33b-9112-4cf2-90b9-de6b42e6072a/c8f23942-50b9-4630-ae9a-ce48e66b7281/"
        "proxy/8000/api/v1"
    )
    tracer = 'https://nat-notebook-inspire.sii.edu.cn/ws-677a4cfa-ac67-494c-b481-e9147a3487a2/project-034ec99f-f57a-4c71-9f7e-1654d2c430c8/user-dc5518ff-a2c5-45ed-a36f-3931cb726d94/vscode/9b87a33b-9112-4cf2-90b9-de6b42e6072a/c8f23942-50b9-4630-ae9a-ce48e66b7281/proxy/8000/api/v1'

    # === Request Handler：必须用 HolosRequestHandler 才能上报 tracing ===
    request_handler = HolosRequestHandler(
        agent_executor=PromptEnhanceAgentExecutor(),   # 你的执行器
        task_store=InMemoryTaskStore(),
        tracer=tracer,
    )

    # === Starlette app ===
    app = A2AStarletteApplication(
        agent_card=public_card,
        http_handler=request_handler,
    ).build()
    app.add_middleware(AccessLog)

    @app.route("/health")
    async def health(_):
        return JSONResponse({"status": "ok", "version": public_card.version})

    uvicorn.run(app, host="0.0.0.0", port=10130, log_level="info")
