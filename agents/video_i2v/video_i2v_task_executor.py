# video_i2v_task_executor.py
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import uuid
import anyio
import logging
import mimetypes
import requests
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional

import dashscope
from dashscope import VideoSynthesis

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_text_artifact
from a2a.types import TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent

from shared.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DEFAULT_SAVE_ARTIFACTS,
    DEFAULT_VIDEO_DURATION,
)

logger = logging.getLogger("video_i2v_executor")


# -----------------------
# Small helpers
# -----------------------
def _json(d: Dict[str, Any]) -> str:
    return json.dumps(d, ensure_ascii=False)

def _status_json(state: str, trace_id: str, payload: Dict[str, Any] | None = None) -> str:
    return _json({
        "type": "video_status",
        "state": state,   # started | generating | downloading | saved_file | succeeded | failed
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
    """优先 parts[*].text；否则 text/content。"""
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

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _guess_ext_from_headers(headers: Dict[str, str]) -> str:
    ct = headers.get("Content-Type") or headers.get("content-type") or ""
    ext = mimetypes.guess_extension(ct.split(";")[0].strip()) if ct else None
    if not ext:
        if "mp4" in ct: ext = ".mp4"
        elif "webm" in ct: ext = ".webm"
    return ext or ".mp4"

def _sanitize_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name or "").strip() or uuid.uuid4().hex

def _make_file_url(local_path: str) -> str:
    local_path = os.path.abspath(local_path)
    return "file://" + local_path

def _pick_first_image_from_dir(dir_path: str) -> Optional[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    for pattern in exts:
        files = sorted(Path(dir_path).glob(pattern))
        if files:
            return str(files[0])
    return None


# -----------------------
# Param extraction
# -----------------------
def _extract_user_params(context: RequestContext) -> Dict[str, Any]:
    """
    输入解析策略：
      - 用户文本若是 JSON，读取其中 prompt/resolution/duration/audio_url/seed/model 等
      - 否则将整段文本作为 prompt
      - 图片固定从后端目录取第一张（可自行在后端改路径/取图策略）
    """
    # 地域切换（默认北京；新加坡请设：DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1）
    dashscope.base_http_api_url = os.getenv("DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL)

    text = _extract_text_from_context(context)
    params = _maybe_json(text) or {}
    if "prompt" not in params or not isinstance(params.get("prompt"), str) or not params["prompt"].strip():
        params["prompt"] = text or "把这张图片转成有趣的短视频。"

    # 固定本地图片路径（后端可自行替换目录/文件）
    FIXED_DIR = os.getenv(
        "I2V_IMAGE_DIR",
        "/inspire/hdd/project/25jinqiu15/haoxiangzhao-P-253130075/agent/source"
    )
    local_image = _pick_first_image_from_dir(FIXED_DIR)
    if not local_image:
        params.setdefault("img_url", "")
    else:
        params["img_url"] = _make_file_url(local_image)

    # 默认参数
    params.setdefault("model", os.getenv("DASHSCOPE_I2V_MODEL", "wan2.5-i2v-preview"))
    params.setdefault("resolution", os.getenv("DASHSCOPE_I2V_RESOLUTION", "480P"))  # 480P/720P/…
    params.setdefault("duration", int(os.getenv("DASHSCOPE_I2V_DURATION", str(DEFAULT_VIDEO_DURATION))))
    params.setdefault("prompt_extend", True)
    params.setdefault("watermark", False)
    params.setdefault("negative_prompt", "")
    # 可选音频：params.setdefault("audio_url", "https://…/xxx.mp3")

    # 保存策略
    params.setdefault("save", DEFAULT_SAVE_ARTIFACTS)
    params.setdefault("save_dir", os.getenv("VIDEO_SAVE_DIR", "outputs/videos"))
    params.setdefault("overwrite", False)

    return params


# -----------------------
# Core I2V Agent
# -----------------------
class VideoI2VCore:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set.")

    def _blocking_call(self, call_kwargs: Dict[str, Any]):
        # DashScope 同步调用
        return VideoSynthesis.call(**call_kwargs)

    def _blocking_download(self, url: str, target: Path, overwrite: bool) -> Path:
        if target.exists() and not overwrite:
            return target
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        if target.suffix == "":
            target = target.with_suffix(_guess_ext_from_headers(r.headers))
        _ensure_dir(target.parent)
        with open(target, "wb") as f:
            f.write(r.content)
        return target

    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img_url = params.get("img_url") or ""
        if not isinstance(img_url, str) or not img_url.strip():
            return {"status_code": 400, "code": "MissingImage", "message": "img_url is required (file:// or http(s)://)."}

        call_kwargs = dict(
            api_key=self.api_key,
            model=params.get("model"),
            prompt=params.get("prompt"),
            img_url=img_url,
            audio_url=params.get("audio_url"),   # 可为 None
            resolution=params.get("resolution", "480P"),
            duration=int(params.get("duration", 8)),
            prompt_extend=bool(params.get("prompt_extend", True)),
            watermark=bool(params.get("watermark", False)),
            negative_prompt=params.get("negative_prompt", ""),
        )
        if params.get("seed") is not None:
            call_kwargs["seed"] = int(params["seed"])

        rsp = await anyio.to_thread.run_sync(self._blocking_call, call_kwargs)

        if rsp.status_code != HTTPStatus.OK:
            return {
                "status_code": int(rsp.status_code),
                "code": getattr(rsp, "code", "DashScopeError"),
                "message": getattr(rsp, "message", "Unknown error from DashScope"),
            }

        video_url = getattr(getattr(rsp, "output", None), "video_url", None)
        return {
            "status_code": 200,
            "model": call_kwargs["model"],
            "request": {k: v for k, v in call_kwargs.items() if k != "api_key"},
            "video_url": video_url,
        }

    async def save(self, video_url: str, save_dir: Path, filename_hint: str = "") -> Dict[str, str]:
        name = _sanitize_name(Path(video_url).stem or filename_hint or uuid.uuid4().hex)
        target = save_dir / f"{name}"
        p = await anyio.to_thread.run_sync(self._blocking_download, video_url, target, False)
        return {"url": video_url, "path": str(p)}


# -----------------------
# A2A Executor
# -----------------------
class VideoI2VTaskAgentExecutor(AgentExecutor):
    """
    图生视频（I2V）执行器：
      事件序列：started -> generating -> (downloading/saved_file) -> video_result -> succeeded
      结果产物：
        - video_params_echo（参数回显，便于前端展示）
        - video_result（紧凑 JSON：model/prompt/image/video_url/local_path/duration/resolution）
        - video_human（人类可读总结：前端直接展示）
    """

    def __init__(self, api_key: Optional[str] = None):
        self.core = VideoI2VCore(api_key=api_key)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 打印 message 便于排查
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

        # 2) working
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.working),
            final=False,
        ))

        # 3) 参数
        params = _extract_user_params(context)

        # 参数回显（给前端一个清晰面板）
        echo = {
            "model": params.get("model"),
            "resolution": params.get("resolution"),
            "duration": params.get("duration"),
            "img_url": params.get("img_url", ""),
            "has_audio": bool(params.get("audio_url")),
        }
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="video_params_echo", text=_json(echo)),
            context_id=context.context_id, task_id=task.id,
        ))

        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="video_status",
                text=_status_json("started", trace_id, {
                    "model": echo["model"], "resolution": echo["resolution"], "duration": echo["duration"]
                })
            ),
            context_id=context.context_id, task_id=task.id,
        ))
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="video_status",
                text=_status_json("generating", trace_id, {
                    "img_url": echo["img_url"],
                    "prompt_preview": (params.get("prompt") or "")[:80],
                })
            ),
            context_id=context.context_id, task_id=task.id,
        ))

        # 4) 调用 I2V
        try:
            result = await self.core.run(params)
        except Exception as e:
            await self._fail(context, event_queue, task.id, trace_id, f"I2V call error: {e}")
            return

        if result.get("status_code") != 200 or not result.get("video_url"):
            await self._fail(
                context, event_queue, task.id, trace_id,
                f"{result.get('code')} - {result.get('message')}"
            )
            return

        video_url = result["video_url"]
        saved: Dict[str, str] = {}

        # 5) 可选保存
        if params.get("save", True):
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="video_status", text=_status_json("downloading", trace_id, {})),
                context_id=context.context_id, task_id=task.id,
            ))
            try:
                save_dir = Path(params.get("save_dir", "outputs/videos"))
                _ensure_dir(save_dir)
                saved = await self.core.save(video_url, save_dir, filename_hint="i2v")
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(
                        name="video_saved",
                        text=_status_json("saved_file", trace_id, {"url": saved.get("url"), "path": saved.get("path")})
                    ),
                    context_id=context.context_id, task_id=task.id,
                ))
            except Exception as e:
                # 保存失败不阻断主流程，仅记录
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(
                        name="video_status",
                        text=_status_json("saved_file", trace_id, {"url": video_url, "error": str(e)})
                    ),
                    context_id=context.context_id, task_id=task.id,
                ))

        # 6) 机读结果（给编排/后处理）
        compact = {
            "traceId": trace_id,
            "model": result.get("model"),
            "prompt": (result.get("request") or {}).get("prompt", ""),
            "image": (result.get("request") or {}).get("img_url", ""),
            "video_url": video_url,
            "local_path": saved.get("path", ""),
            "resolution": (result.get("request") or {}).get("resolution", ""),
            "duration": (result.get("request") or {}).get("duration", 0),
            "audio_url": (result.get("request") or {}).get("audio_url", None),
        }
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="video_result", text=_json(compact)),
            context_id=context.context_id, task_id=task.id,
        ))

        # 7) 人类可读总结（给前端直接展示）
        human = (
            "✅ 图生视频完成\n"
            f"模型: {compact['model']}\n"
            f"分辨率: {compact['resolution']}  时长: {compact['duration']}s\n"
            f"图片: {compact['image']}\n"
            f"视频链接: {compact['video_url']}\n"
            + (f"本地保存: {compact['local_path']}\n" if compact['local_path'] else "")
        )
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="video_human", text=human),
            context_id=context.context_id, task_id=task.id,
        ))

        # 8) 成功 & completed
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="video_status", text=_status_json("succeeded", trace_id, {"video_url": video_url})),
            context_id=context.context_id, task_id=task.id,
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ))

    async def _fail(self, context: RequestContext, eq: EventQueue, task_id: str, trace_id: str, msg: str):
        logger.error("[trace:%s] FAILED | %s", trace_id, msg)
        await eq.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="video_error", text=f"[trace:{trace_id}] ❌ {msg}"),
            context_id=context.context_id, task_id=task_id,
        ))
        await eq.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="video_status", text=_status_json("failed", trace_id, {"error": msg})),
            context_id=context.context_id, task_id=task_id,
        ))
        await eq.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task_id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = getattr(context, "task_id", None) or "unknown"
        canceled_state = getattr(TaskState, "canceled", TaskState.completed)
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="video_status", text=_status_json("failed", uuid.uuid4().hex[:8], {"canceled": True})),
            context_id=context.context_id, task_id=task_id,
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task_id,
            status=TaskStatus(state=canceled_state),
            final=True,
        ))
