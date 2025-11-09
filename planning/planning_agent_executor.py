# planning/planning_agent_executor.py
# -*- coding: utf-8 -*-
"""Planning agent that orchestrates downstream task agents."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional

import anyio
import httpx
from openai import AzureOpenAI

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_text_artifact
from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent, TaskArtifactUpdateEvent

from holos_sdk import HolosA2AClientFactory, Plan, PlantTracer
from a2a.client.client import ClientConfig
from a2a.client.card_resolver import A2ACardResolver

from shared.config import (
    ROUTER_PUBLIC_URL,
    YUNSTORM_API_KEY,
    YUNSTORM_API_VERSION,
    YUNSTORM_ENDPOINT,
    YUNSTORM_MODEL,
)
from shared.logger import setup_logging

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
    if not isinstance(steps, list):
        raise ValueError("Plan JSON must contain a list under 'steps'.")

    if not steps:
        root = Plan(goal="Prompt orchestration")
        root.metadata = {"op": "noop", "params": {}}
        return root

    id2plan: Dict[str, Plan] = {}
    for raw in steps:
        if not isinstance(raw, dict):
            continue
        sid = (raw.get("id") or uuid.uuid4().hex[:6]).strip()
        op = (raw.get("op") or "").strip()
        params = raw.get("params") or {}
        prompt_preview = params.get("prompt") or params.get("image_url") or params.get("img_url") or ""
        goal = f"{op.upper()} | {prompt_preview}".strip() or op or "task"

        node = Plan(goal=goal)
        node.metadata = {"op": op, "params": params}
        id2plan[sid] = node

    for raw in steps:
        sid = raw.get("id") if isinstance(raw, dict) else None
        if not sid or sid not in id2plan:
            continue
        dependencies = []
        for dep in raw.get("after", []) if isinstance(raw, dict) else []:
            if dep in id2plan:
                dependencies.append(id2plan[dep])
        if dependencies:
            id2plan[sid].depend_plans = dependencies

    referenced = {dep for raw in steps if isinstance(raw, dict) for dep in raw.get("after", [])}
    roots = [id2plan[raw.get("id")] for raw in steps if isinstance(raw, dict) and raw.get("id") in id2plan and raw.get("id") not in referenced]
    root = (roots or list(id2plan.values()))[0]

    meta = dict(getattr(root, "metadata", {}) or {})
    if "notes" in llm_plan:
        meta["notes"] = llm_plan["notes"]
    root.metadata = meta
    return root

class PlanningAgentExecutor(AgentExecutor):
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.llm  = YunstormLLM()
        self.ROUTER_AGENT_URL = os.getenv("ROUTER_URL", ROUTER_PUBLIC_URL)

    # ------------------------------------------------------------------ helpers
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

    def _extract_user_text(self, context: RequestContext) -> str:
        """Best-effort extraction of the user's textual instruction."""

        def _iter_candidates() -> Iterable[Any]:
            for path in [
                ("message",),
                ("params", "message"),
                ("request", "message"),
                ("input", "message"),
                ("input_message",),
            ]:
                current: Any = context
                ok = True
                for key in path:
                    if isinstance(current, dict):
                        current = current.get(key)
                    else:
                        current = getattr(current, key, None)
                    if current is None:
                        ok = False
                        break
                if ok and current is not None:
                    yield current

        def _text_from_message(msg: Any) -> str:
            if msg is None:
                return ""
            # Structured message objects from `a2a` expose `.parts` with `.root.text`.
            parts = getattr(msg, "parts", None)
            if isinstance(parts, list):
                for part in parts:
                    root = getattr(part, "root", None)
                    text = getattr(root, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text.strip()

            if hasattr(msg, "model_dump"):
                msg = msg.model_dump(exclude_none=True)

            if isinstance(msg, dict):
                for part in msg.get("parts", []):
                    text = part.get("text") if isinstance(part, dict) else None
                    if isinstance(text, str) and text.strip():
                        return text.strip()
                for key in ("text", "content"):
                    text = msg.get(key)
                    if isinstance(text, str) and text.strip():
                        return text.strip()

            if isinstance(msg, str) and msg.strip():
                return msg.strip()

            return ""

        for candidate in _iter_candidates():
            text = _text_from_message(candidate)
            if text:
                return text
        return ""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = new_task(context.message); task.id = context.task_id
        await event_queue.enqueue_event(task)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.working), final=False,
        ))

        # 提取用户文本
        user_text = self._extract_user_text(context) or "生成一个可爱的视觉作品"

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
