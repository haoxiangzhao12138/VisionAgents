# planning/planning_agent_executor.py
# -*- coding: utf-8 -*-
import os, re, json, time, uuid, anyio, httpx
from typing import Any, Dict, List, Optional
from openai import AzureOpenAI

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_text_artifact
from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent, TaskArtifactUpdateEvent

from holos_sdk import HolosA2AClientFactory, Plan, PlantTracer
from a2a.client.client import ClientConfig
from a2a.client.card_resolver import A2ACardResolver

from shared.logger import setup_logging
from shared.config import (
    YUNSTORM_ENDPOINT, YUNSTORM_API_KEY, YUNSTORM_API_VERSION, YUNSTORM_MODEL,
)

logger = setup_logging("planning_agent")

class YunstormLLM:
    def __init__(self):
        if not YUNSTORM_API_KEY:
            raise RuntimeError("YUNSTORM_API_KEY is not set.")
        self.client = AzureOpenAI(
            azure_endpoint=YUNSTORM_ENDPOINT,
            api_key=YUNSTORM_API_KEY,
            api_version=YUNSTORM_API_VERSION,
        )
        self.model = YUNSTORM_MODEL

    async def plan(self, system_prompt: str, user_text: str) -> dict:
        def _blocking():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_text}],
                temperature=0.2,
                response_format={"type":"json_object"},
                max_tokens=800,
            )
        resp = await anyio.to_thread.run_sync(_blocking)
        text = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                raise RuntimeError(f"LLM returned non-JSON: {text[:200]}")
            return json.loads(m.group(0))

def build_plan_tree(llm_plan: Dict[str, Any]) -> Plan:
    steps = llm_plan.get("steps") or []
    id2plan: Dict[str, Plan] = {}
    for s in steps:
        sid = s.get("id") or uuid.uuid4().hex[:6]
        op  = (s.get("op") or "").strip()
        prm = s.get("params") or {}
        goal = f"{op.upper()} | {prm.get('prompt') or prm.get('image_url') or prm.get('img_url') or ''}".strip()
        p = Plan(goal=goal)
        p.metadata = {"op": op, "params": prm}
        id2plan[sid] = p
    # 依赖
    for s in steps:
        sid = s.get("id")
        after = s.get("after") or []
        id2plan[sid].depend_plans = [id2plan[a] for a in after if a in id2plan]
    # 根
    referenced = {a for s in steps for a in (s.get("after") or [])}
    roots = [id2plan[s.get("id")] for s in steps if s.get("id") not in referenced] or list(id2plan.values())[:1]
    root = roots[0]
    if "notes" in llm_plan:
        root.metadata = (root.metadata or {}) | {"notes": llm_plan["notes"]}
    return root

class PlanningAgentExecutor(AgentExecutor):
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.llm  = YunstormLLM()
        self.ROUTER_AGENT_URL = os.getenv("ROUTER_URL", "http://localhost:10102/")

    def _planner_system_prompt(self) -> str:
        return (
            "You are a senior visual orchestration planner.\n"
            "Decompose the user requirement into atomic ops among: [prompt_boost, t2i, t2v, i2v, edit].\n"
            "Rules:\n"
            "1) If user prompt is short/generic (<= 8 tokens) or lacks key visual attributes, ADD a `prompt_boost` step first.\n"
            "2) Always give downstream steps an explicit prompt. If `prompt_boost` exists, set prompt as {'$ref':'<boostId>.enhanced_prompt'}.\n"
            "3) URL-only pipeline: downstream I2V that consumes an image MUST reference previous T2I result URL via {'$ref':'<t2iId>.results.0.url'}.\n"
            "4) Defaults: image 1024*1024; video 832*480; duration 6-10s.\n"
            "5) STRICT JSON: {steps:[{id, op, params, after:[] }], notes}.\n"
            "6) Keep the user's language.\n"
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = new_task(context.message); task.id = context.task_id
        await event_queue.enqueue_event(task)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.working), final=False,
        ))

        # 提取用户文本
        user_text = ""
        try:
            for p in context.message.parts:  # type: ignore
                if isinstance(p, dict) and p.get("kind")=="text" and (p.get("text") or "").strip():
                    user_text = p["text"].strip(); break
        except Exception:
            pass
        user_text = user_text or "生成一个可爱的视觉作品"

        # 调 LLM 生成 plan
        try:
            llm_plan = await self.llm.plan(self._planner_system_prompt(), user_text)
        except Exception as e:
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="plan_error", text=f"❌ LLM plan error: {e}"),
                context_id=context.context_id, task_id=task.id,
            ))
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                context_id=context.context_id, task_id=task.id,
                status=TaskStatus(state=TaskState.completed), final=True,
            ))
            return

        # 回显 JSON & Plan
        plan = build_plan_tree(llm_plan)
        await event_queue.enqueue_event(plan)
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="plan_json", text=json.dumps(llm_plan, ensure_ascii=False, indent=2)),
            context_id=context.context_id, task_id=task.id,
        ))

        # 透传到 Router（send_plan_streaming）
        try:
            tracer: PlantTracer = context.call_context.state["tracer"]  # type: ignore
            resolver = A2ACardResolver(httpx_client=self.http, base_url=self.ROUTER_AGENT_URL)
            router_card = await resolver.get_agent_card()
            factory = HolosA2AClientFactory(ClientConfig(httpx_client=self.http), tracer=tracer)
            router_client = factory.create(router_card)
            async for evt in router_client.send_plan_streaming(plan):
                await event_queue.enqueue_event(evt)
        except Exception as e:
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="plan_error", text=f"❌ Route failed: {e}"),
                context_id=context.context_id, task_id=task.id,
            ))

        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.completed), final=True,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        canceled_state = getattr(TaskState, "canceled", TaskState.completed)
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="plan_status", text=json.dumps({"type":"plan_status","state":"failed","canceled":True}, ensure_ascii=False)),
            context_id=context.context_id, task_id=getattr(context,"task_id","unknown"),
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=getattr(context,"task_id","unknown"),
            status=TaskStatus(state=canceled_state), final=True,
        ))
