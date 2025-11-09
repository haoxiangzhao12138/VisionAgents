# prompt_enhance_executor.py
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import uuid
import anyio
import logging
from typing import Any, Dict, Optional, Tuple

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_text_artifact
from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent, TaskArtifactUpdateEvent

# =========================
# 云风 GPT 网关 (AzureOpenAI 兼容)
# =========================
# 文档要求：
#   azure_endpoint="https://gpt.yunstorm.com/"
#   api_version="2025-04-01-preview"
#   api_key="$API_KEY"
from openai import AzureOpenAI  # pip install openai

logger = logging.getLogger("prompt_enhance_executor")

# ====== 配置（可用环境变量覆盖）======
YUNSTORM_ENDPOINT = os.getenv("YUNSTORM_ENDPOINT", "https://gpt.yunstorm.com/")
YUNSTORM_API_KEY = "c1660c7c06c32f4a48c5bac00e5852a5"
YUNSTORM_API_VERSION = os.getenv("YUNSTORM_API_VERSION", "2025-04-01-preview")

DEFAULT_MODEL = os.getenv("PROMPT_ENHANCE_MODEL", "gpt-4.1")
DEFAULT_TEMPERATURE = float(os.getenv("PROMPT_ENHANCE_TEMPERATURE", "0.7"))

SYSTEM_PROMPT = """You are a Prompt Enhancer specialized for visual tasks (image editing, text-to-image, text-to-video, image-to-video).
Your job:
1) Read the user's task and optional JSON params.
2) Improve the prompt: make it vivid, specific, and actionable for a generative model (DashScope/wan2.5, qwen-image-edit-plus, T2V/I2V).
3) Respect constraints: style, size, duration, SFW. Remove unclear references; avoid personal data and disallowed content.
4) Output **two** artifacts:
   A) ENHANCED_PROMPT (plain text) — concise but expressive, ready for downstream model.
   B) STRUCTURED_JSON (JSON) — include fields: task_type, enhanced_prompt, key_visual_elements, style, composition, camera_or_motion, quality_tags, negative_prompt, safety_notes.

Guidelines:
- If the user text is Chinese, respond in Chinese; else respond in English.
- Keep the enhanced prompt <= 180 words (or <= 200 Chinese characters) unless absolutely necessary.
- For image editing with 1~3 images: refer to them as 图1/图2/图3 (ZH) or Image#1/#2/#3 (EN). Do NOT invent image content.
- For T2V/I2V: include motion, pacing, camera moves (pan/tilt/dolly), duration hint (non-binding).
- For T2I: specify objects, materials, lighting, palette, lens/camera, time of day, mood.
- Avoid copyrighted names; use generic descriptions instead.

Output format (no extra prose):
<ENHANCED_PROMPT>
---
<STRUCTURED_JSON>
"""

# =========================
# Helpers
# =========================

_SPLIT_RE = re.compile(r"\n-{3,}\n|-{3,}", re.MULTILINE)

def _strip_tag(s: str) -> str:
    """去掉标签与代码围栏"""
    s = (s or "").strip()
    s = re.sub(r"^<ENHANCED_PROMPT>\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^<STRUCTURED_JSON>\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^```.*?\n", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"\n```$", "", s).strip()
    return s

def _parse_two_block_output(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    期望模型输出：
    <ENHANCED_PROMPT>
    ...
    ---
    <STRUCTURED_JSON>
    {...}
    """
    if not isinstance(text, str):
        return "", {}

    parts = _SPLIT_RE.split(text, maxsplit=1)
    if len(parts) == 2:
        enhanced_raw, struct_raw = parts[0], parts[1]
        enhanced = _strip_tag(enhanced_raw)
        struct_s = _strip_tag(struct_raw)
        # 尝试解析 JSON
        try:
            structured = json.loads(struct_s)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", struct_s)
            if m:
                try:
                    structured = json.loads(m.group(0))
                except Exception:
                    structured = {}
            else:
                structured = {}
        return enhanced, structured

    # 没有 ---：退化为整段当增强文本，再尽力找 JSON
    cleaned = _strip_tag(text)
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        struct_txt = m.group(0)
        try:
            structured = json.loads(struct_txt)
            enhanced = cleaned.replace(struct_txt, "").strip()
            return enhanced, structured
        except Exception:
            pass
    return cleaned, {}

def _json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)

def _status_json(state: str, trace_id: str, payload: Dict[str, Any] | None = None) -> str:
    return _json({
        "type": "prompt_enhance_status",
        "state": state,  # started | succeeded | failed
        "traceId": trace_id,
        "timestamp": time.time(),
        "data": payload or {},
    })

def _safe_get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _to_dict(obj: Any) -> Any:
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_none=True)
    except Exception:
        pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
        except Exception:
            return obj
    return obj

def _extract_text_from_context(context: RequestContext) -> str:
    """兼容多种 message 形状：优先 parts[*].text，然后 text/content。"""
    for path in [
        ("message",),
        ("params", "message"),
        ("request", "message"),
        ("input", "message"),
        ("input_message",),
    ]:
        cur = context
        ok = True
        for k in path:
            cur = _safe_get(cur, k, None)
            if cur is None:
                ok = False
                break
        if not ok or cur is None:
            continue

        cur = _to_dict(cur)

        parts = _safe_get(cur, "parts", [])
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and (p.get("kind") == "text" or p.get("type") == "text"):
                    t = (p.get("text") or "").strip()
                    if t:
                        return t

        for key in ("text", "content"):
            t = _safe_get(cur, key, "")
            if isinstance(t, str) and t.strip():
                return t.strip()
    return ""

def _maybe_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _detect_lang(s: str) -> str:
    """简单语言检测：中文占比>20%即视为中文。"""
    if not s:
        return "zh"
    cn = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return "zh" if (cn / max(1, len(s)) > 0.2) else "en"

def _build_user_prompt(raw_text: str, parsed: Optional[Dict[str, Any]], lang: str) -> str:
    """
    将用户输入打包给模型；支持 task_type/size/duration/negative_prompt/style_hint/target_model。
    """
    task_type = parsed.get("task_type") if isinstance(parsed, dict) else None
    target = parsed.get("target_model") if isinstance(parsed, dict) else None

    hint_lines = []
    if task_type: hint_lines.append(f"task_type: {task_type}")
    if target:    hint_lines.append(f"target_model: {target}")
    for k in ("size", "duration", "negative_prompt", "style_hint"):
        v = parsed.get(k) if isinstance(parsed, dict) else None
        if v:
            hint_lines.append(f"{k}: {v}")

    hints = "\n".join(hint_lines) or "(none)"
    header = "用户输入如下：" if lang == "zh" else "User input:"
    return f"{header}\n{raw_text}\n\nHints:\n{hints}"

# =========================
# Core (Yunstorm AzureOpenAI)
# =========================
class PromptEnhanceCore:
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        if not YUNSTORM_API_KEY:
            raise RuntimeError("YUNSTORM_API_KEY / API_KEY is not set.")
        self.client = AzureOpenAI(
            azure_endpoint=YUNSTORM_ENDPOINT,
            api_key=YUNSTORM_API_KEY,
            api_version=YUNSTORM_API_VERSION,
        )
        self.model = model or DEFAULT_MODEL
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)

    async def enhance(self, raw_text: str, parsed_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        lang = _detect_lang(raw_text or json.dumps(parsed_json or {}, ensure_ascii=False))
        user_prompt = _build_user_prompt(raw_text, parsed_json, lang)

        def _blocking():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=800,
            )

        resp = await anyio.to_thread.run_sync(_blocking)
        text = resp.choices[0].message.content or ""

        # 解析为两个块：增强文本 + 结构化 JSON
        enhanced_prompt, structured = _parse_two_block_output(text)
        return {
            "status": "ok",
            "model": self.model,
            "temperature": self.temperature,
            "lang": lang,
            "enhanced_prompt": enhanced_prompt,
            "structured": structured,
        }

# =========================
# Executor（精简事件版）
# =========================
class PromptEnhanceAgentExecutor(AgentExecutor):
    """
    Prompt 增强执行器（前置于图像/视频任务）：
      - 输入：纯文本或 JSON（可包含 task_type/size/duration/negative_prompt/style_hint/target_model）
      - 输出：增强后的纯文本 prompt + 结构化 JSON
      - 仅发送关键事件：started / succeeded + 两个最终产物
    """

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        self.core = PromptEnhanceCore(model=model, temperature=temperature)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 打印原始 message（方便排查）
        try:
            msg = getattr(context, "message", None)
            dump = msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else (msg.__dict__ if hasattr(msg, "__dict__") else str(msg))
            logger.info("[debug] incoming message: %s", json.dumps(dump, ensure_ascii=False))
        except Exception as e:
            logger.warning("[debug] cannot dump message: %s", e)

        trace_id = uuid.uuid4().hex[:8]

        # 1) new_task
        task = new_task(context.message)
        task.id = context.task_id
        await event_queue.enqueue_event(task)

        # 2) working(final=False)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.working),
            final=False,
        ))

        # 3) 取输入文本/JSON（空则兜底）
        raw_text = _extract_text_from_context(context)
        parsed_json = _maybe_json(raw_text)
        if not raw_text.strip():
            raw_text = "生成一段简洁明确的视觉任务提示，主题自拟（演示占位）。"

        # —— 只发 started
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="prompt_enhance_status",
                text=_status_json("started", trace_id, {"has_json": bool(parsed_json)})
            ),
            context_id=context.context_id, task_id=task.id,
        ))

        # 4) 调模型
        t0 = time.perf_counter()
        try:
            result = await self.core.enhance(raw_text, parsed_json)
        except Exception as e:
            await self._fail(context, event_queue, task.id, trace_id, f"Yunstorm GPT error: {e}")
            return
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # 5) 只发两个最终产物
        enhanced_prompt = (result.get("enhanced_prompt") or "").strip()
        structured = result.get("structured") or {}

        # (1) 纯文本增强结果
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="prompt_enhance_text", text=enhanced_prompt),
            context_id=context.context_id, task_id=task.id,
        ))

        # (2) 紧凑 JSON 结果
        compact = {"enhanced_prompt": enhanced_prompt, "structured": structured}
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="prompt_enhance_result", text=json.dumps(compact, ensure_ascii=False)),
            context_id=context.context_id, task_id=task.id,
        ))

        # —— 只发 succeeded
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="prompt_enhance_status",
                text=_status_json("succeeded", trace_id, {"latency_ms": latency_ms})
            ),
            context_id=context.context_id, task_id=task.id,
        ))

        # 6) completed(final=True)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ))

    async def _fail(self, context: RequestContext, queue: EventQueue, task_id: str, trace_id: str, msg: str):
        logger.error("[trace:%s] FAILED | %s", trace_id, msg)
        await queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="prompt_enhance_error", text=f"[trace:{trace_id}] ❌ {msg}"),
            context_id=context.context_id, task_id=task_id,
        ))
        await queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="prompt_enhance_status", text=_status_json("failed", trace_id, {"error": msg})),
            context_id=context.context_id, task_id=task_id,
        ))
        await queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task_id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = getattr(context, "task_id", None) or "unknown"
        canceled_state = getattr(TaskState, "canceled", TaskState.completed)
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="prompt_enhance_status",
                text=_status_json("failed", uuid.uuid4().hex[:8], {"canceled": True})
            ),
            context_id=context.context_id, task_id=task_id,
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task_id,
            status=TaskStatus(state=canceled_state),
            final=True,
        ))
