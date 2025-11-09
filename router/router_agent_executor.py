# -*- coding: utf-8 -*-
import os, json, uuid, logging, httpx, anyio
from typing import Dict, Any, List
from collections import deque, defaultdict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_text_artifact
from a2a.types import (
    TaskState, TaskStatus, TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent, Message, Role, Part, TextPart
)

from holos_sdk import HolosA2AClientFactory, Assignment
from holos_sdk.utils import try_convert_to_plan
from a2a.client.client import ClientConfig
from a2a.client.card_resolver import A2ACardResolver

from shared.config import IMAGE_T2I_URL, VIDEO_T2V_URL, IMAGE_EDIT_URL, VIDEO_I2V_URL, PROMPT_URL

logger = logging.getLogger("router_agent")


def _new_text_message(text: str) -> Message:
    return Message(message_id=uuid.uuid4().hex, role=Role.user, parts=[Part(root=TextPart(text=text))])


def _collect_nodes(root) -> List[Any]:
    stack, seen, result = [root], set(), []
    while stack:
        p = stack.pop()
        pid = getattr(p, "id", None)
        if not pid or pid in seen:
            continue
        seen.add(pid); result.append(p)
        for d in getattr(p, "depend_plans", []) or []:
            stack.append(d)
    return result


def _topo(plans_root) -> List[Any]:
    all_nodes = _collect_nodes(plans_root)
    indeg, graph, nodes = defaultdict(int), defaultdict(list), {n.id: n for n in all_nodes}
    for n in all_nodes:
        for d in getattr(n, "depend_plans", []) or []:
            if getattr(d, "id", None):
                graph[d.id].append(n.id)
                indeg[n.id] += 1
    q = deque([n.id for n in all_nodes if indeg[n.id] == 0])
    order_ids = []
    while q:
        x = q.popleft(); order_ids.append(x)
        for y in graph[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)
    if len(order_ids) != len(all_nodes):
        logger.warning("Router topo: cycle or missing nodes detected; fallback to DFS order.")
        return all_nodes
    return [nodes[_id] for _id in order_ids]


class RouterAgentExecutor(AgentExecutor):
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=60.0))
        self.URL_T2I, self.URL_T2V, self.URL_I2V, self.URL_EDIT, self.URL_PROMP = (
            IMAGE_T2I_URL, VIDEO_T2V_URL, VIDEO_I2V_URL, IMAGE_EDIT_URL, PROMPT_URL
        )

    def __del__(self):
        try:
            if not self.http.is_closed:
                anyio.from_thread.run(self.http.aclose)
        except Exception:
            pass

    async def _resolve_client(self, tracer, base_url: str):
        resolver = A2ACardResolver(httpx_client=self.http, base_url=base_url)
        card = await resolver.get_agent_card()
        factory = HolosA2AClientFactory(ClientConfig(httpx_client=self.http), tracer=tracer)
        return factory.create(card), card

    def _pick_agent_by_op(self, op: str) -> str:
        op = (op or "").lower()
        if op == "t2i": return self.URL_T2I
        if op == "t2v": return self.URL_T2V
        if op == "i2v": return self.URL_I2V
        if op == "edit": return self.URL_EDIT
        if op == "prompt_boost": return self.URL_PROMP
        return ""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = new_task(context.message); task.id = context.task_id
        await event_queue.enqueue_event(task)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.working), final=False,
        ))

        plan = try_convert_to_plan(context.message)
        if not plan:
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="router_error", text="❌ Expect a Holos Plan from Planning Agent."),
                context_id=context.context_id, task_id=task.id,
            ))
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                context_id=context.context_id, task_id=task.id,
                status=TaskStatus(state=TaskState.completed), final=True,
            ))
            return

        try:
            tracer = context.call_context.state["tracer"]
        except Exception:
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="router_error", text="❌ tracer missing in context.call_context.state"),
                context_id=context.context_id, task_id=task.id,
            ))
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                context_id=context.context_id, task_id=task.id,
                status=TaskStatus(state=TaskState.completed), final=True,
            ))
            return

        ordered = _topo(plan)

        id2client: Dict[str, Any] = {}
        for sub in ordered:
            meta = getattr(sub, "metadata", {}) or {}
            op   = (meta.get("op") or "").lower()
            params = meta.get("params") or {}

            base_url = self._pick_agent_by_op(op)
            if not base_url:
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(name="router_warn", text=f"⚠️ Skip unknown op: {op} | goal={getattr(sub,'goal','')}"),
                    context_id=context.context_id, task_id=task.id,
                ))
                continue

            try:
                client, card = id2client.get(base_url) or await self._resolve_client(tracer, base_url)
                id2client[base_url] = (client, card)
                assignment = Assignment(object_id=sub.id, assignee_id=card.url, assignee_name=card.name)
                await event_queue.enqueue_event(assignment)
            except Exception as e:
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(name="router_error", text=f"❌ Resolve agent failed: {e}"),
                    context_id=context.context_id, task_id=task.id,
                ))
                continue

            try:
                msg = _new_text_message(json.dumps(params, ensure_ascii=False))
                async for evt in client.send_message(msg, from_objects=[sub.id]):
                    await event_queue.enqueue_event(evt)
            except Exception as e:
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(name="router_error", text=f"❌ Dispatch op={op} failed: {e}"),
                    context_id=context.context_id, task_id=task.id,
                ))

        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="router_status", text="✅ Routing completed."),
            context_id=context.context_id, task_id=task.id,
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.completed), final=True,
        ))