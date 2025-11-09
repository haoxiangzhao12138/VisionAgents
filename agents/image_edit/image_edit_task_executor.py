# image_edit_task_executor.py
# -*- coding: utf-8 -*-
import os, re, json, time, uuid, glob, anyio, logging, mimetypes, requests
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional

import dashscope
from dashscope import MultiModalConversation

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_text_artifact
from a2a.types import TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent

from shared.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DEFAULT_SAVE_ARTIFACTS

# ---------- Config ----------
FIXED_DIR = os.getenv("FIXED_IMAGE_DIR", "/inspire/hdd/project/25jinqiu15/haoxiangzhao-P-253130075/agent/outputs")
DEFAULT_EDIT_INSTRUCTION = os.getenv("FIXED_EDIT_INSTRUCTION", "合并几张图片")
dashscope.base_http_api_url = os.getenv("DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL)

logger = logging.getLogger("image_edit_executor")

# ---------- helpers ----------
def _status_json(state: str, trace_id: str, payload: Dict[str, Any] | None = None) -> str:
    return json.dumps({"type": "image_edit_status", "state": state, "traceId": trace_id, "timestamp": time.time(), "data": payload or {}}, ensure_ascii=False)

def _sanitize_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", s or "").strip() or uuid.uuid4().hex

def _guess_ext_from_headers(headers: Dict[str, str]) -> str:
    ct = headers.get("Content-Type") or headers.get("content-type") or ""
    ext = mimetypes.guess_extension(ct.split(";")[0].strip()) if ct else None
    if not ext:
        if "image/png" in ct: ext = ".png"
        elif "image/jpeg" in ct or "image/jpg" in ct: ext = ".jpg"
        elif "image/webp" in ct: ext = ".webp"
        elif "image/gif" in ct: ext = ".gif"
    return ext or ".png"

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _collect_text_from_message(context: RequestContext) -> str:
    """尽量兼容：优先 parts[*].text；再回退 text/content。"""
    try:
        msg = getattr(context, "message", None)
        if not msg: return ""
        # 强类型读取
        parts = getattr(msg, "parts", None)
        if parts:
            for p in parts:
                root = getattr(p, "root", None)
                text = getattr(root, "text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()
        # 回退 dict
        if hasattr(msg, "model_dump"):
            d = msg.model_dump(exclude_none=True)
            for p in d.get("parts", []):
                if p.get("kind") == "text" and p.get("text"):
                    return p["text"].strip()
            if isinstance(d.get("text"), str) and d["text"].strip():
                return d["text"].strip()
    except Exception:
        pass
    return ""

def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s: return None
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return None
    return None

def _pick_first_images_from_dir(dir_path: str, limit: int = 3) -> list[str]:
    files: list[str] = []
    for pat in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        files.extend(glob.glob(os.path.join(dir_path, pat)))
    files.sort()
    return files[:limit]

# ---------- param parsing ----------
def _extract_user_params(context: RequestContext) -> Dict[str, Any]:
    """
    优先使用上游 JSON 参数；缺失再回退默认。
    可用字段：instruction, images(list[str]), n, watermark, negative_prompt, model, save, save_dir, overwrite
    """
    text = _collect_text_from_message(context)
    params: Dict[str, Any] = _try_parse_json(text) or {}

    # instruction：优先 JSON；否则用纯文本；再兜底 DEFAULT_EDIT_INSTRUCTION
    if not isinstance(params.get("instruction"), str) or not params["instruction"].strip():
        params["instruction"] = text.strip() if isinstance(text, str) and text.strip() else DEFAULT_EDIT_INSTRUCTION

    # images：优先 JSON；否则用 FIXED_DIR 拾取
    if not isinstance(params.get("images"), list) or not params["images"]:
        params["images"] = _pick_first_images_from_dir(FIXED_DIR, limit=3)

    # 默认值
    params.setdefault("n", 1)
    params.setdefault("negative_prompt", "")
    params.setdefault("watermark", False)
    params.setdefault("model", os.getenv("DASHSCOPE_EDIT_MODEL", "qwen-image-edit-plus"))
    params.setdefault("save", DEFAULT_SAVE_ARTIFACTS)
    params.setdefault("save_dir", os.getenv("IMAGE_EDIT_SAVE_DIR", "outputs/edits"))
    params.setdefault("overwrite", False)
    return params

# ---------- core ----------
class ImageEditCore:
    """封装 DashScope 多模态编辑（qwen-image-edit-plus）。"""

    def __init__(self, api_key: Optional[str] = None):
        # 不在构造期抛错，避免服务启动失败；缺失密钥时在调用处返回错误事件
        self.api_key = api_key or DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")

    def _blocking_call(self, messages: List[Dict[str, Any]], n: int, watermark: bool, negative_prompt: str, model: str):
        return MultiModalConversation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
            stream=False,
            n=int(n),
            watermark=bool(watermark),
            negative_prompt=negative_prompt or " ",
        )

    def _blocking_download(self, url: str, target: Path, overwrite: bool) -> Path:
        if target.exists() and not overwrite:
            return target
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if target.suffix == "":
            target = target.with_suffix(_guess_ext_from_headers(r.headers))
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            f.write(r.content)
        return target

    async def edit(self, images: List[str], instruction: str, n: int, watermark: bool,
                   negative_prompt: str, model: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"status_code": 400, "code": "MissingAPIKey", "message": "DASHSCOPE_API_KEY is not set."}
        if not isinstance(instruction, str) or not instruction.strip():
            return {"status_code": 400, "code": "MissingInstruction", "message": "instruction is required"}

        content: List[Dict[str, str]] = []
        for img in images[:3]:
            if isinstance(img, str) and img.strip():
                content.append({"image": img})
        content.append({"text": instruction.strip()})

        messages = [{"role": "user", "content": content}]
        rsp = await anyio.to_thread.run_sync(self._blocking_call, messages, n, watermark, negative_prompt, model)

        if rsp.status_code != HTTPStatus.OK:
            return {"status_code": int(rsp.status_code), "code": getattr(rsp, "code", "DashScopeError"), "message": getattr(rsp, "message", "Unknown error from DashScope")}

        outputs: List[str] = []
        try:
            for item in rsp.output.choices[0].message.content:
                url = item.get("image")
                if isinstance(url, str) and url.strip():
                    outputs.append(url.strip())
        except Exception:
            pass

        return {
            "status_code": 200,
            "model": model,
            "request": {"n": int(n), "watermark": bool(watermark), "negative_prompt": negative_prompt, "images": images, "instruction": instruction},
            "results": outputs,
        }

    async def save_all(self, urls: List[str], save_dir: Path, prefix: str = "edit", overwrite: bool = False) -> List[Dict[str, str]]:
        _ensure_dir(save_dir)
        saved: List[Dict[str, str]] = []

        async def _one(i: int, u: str):
            name = _sanitize_name(Path(u).stem or f"{prefix}_{i}")
            target = save_dir / f"{name}_{i}"
            try:
                p = await anyio.to_thread.run_sync(self._blocking_download, u, target, overwrite)
                saved.append({"url": u, "path": str(p)})
            except Exception as e:
                saved.append({"url": u, "path": "", "error": str(e)})

        async with anyio.create_task_group() as tg:
            for i, u in enumerate(urls):
                tg.start_soon(_one, i, u)
        return saved

# ---------- executor ----------
class ImageEditTaskAgentExecutor(AgentExecutor):
    """
    new_task → working → progress(flow) → result/error → completed(final=True)
    progress: started → validating → calling_model → model_done → downloading/saved_file* → succeeded/failed
    """
    def __init__(self, api_key: Optional[str] = None):
        self.core = ImageEditCore(api_key=api_key)

    async def _emit_progress(self, event_queue: EventQueue, context: RequestContext, task_id: str, trace_id: str,
                             stage: str, percent: int, extra: Optional[Dict[str, Any]] = None,
                             artifact_name: str = "image_edit_status", log: bool = True) -> None:
        payload = {"stage": stage, "percent": max(0, min(100, int(percent)))}
        if extra: payload.update(extra)
        if log: logger.info("[trace:%s] %-12s | %3d%% | %s", trace_id, stage, payload["percent"], extra or "")
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name=artifact_name, text=_status_json(stage, trace_id, payload)),
            context_id=context.context_id, task_id=task_id,
        ))

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        trace_id = uuid.uuid4().hex[:8]

        # new_task
        task = new_task(context.message); task.id = context.task_id
        await event_queue.enqueue_event(task)

        # working
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.working), final=False,
        ))

        # params & started
        params = _extract_user_params(context)
        await self._emit_progress(event_queue, context, task.id, trace_id, "started", 0,
                                  {"images": params.get("images", []), "n": params.get("n", 1),
                                   "model": params.get("model"), "instruction_preview": (params.get("instruction","")[:80])})

        # validating
        await self._emit_progress(event_queue, context, task.id, trace_id, "validating", 5)
        if not params.get("images"):
            await self._fail(context, event_queue, task.id, trace_id, "No images provided (nor found in FIXED_DIR)")
            return

        # call model
        await self._emit_progress(event_queue, context, task.id, trace_id, "calling_model", 10,
                                  {"n": params.get("n", 1), "watermark": params.get("watermark", False)})
        t0 = time.perf_counter()
        try:
            result = await self.core.edit(
                images=params["images"],
                instruction=params["instruction"],
                n=int(params.get("n", 1)),
                watermark=bool(params.get("watermark", False)),
                negative_prompt=params.get("negative_prompt", ""),
                model=params.get("model", "qwen-image-edit-plus"),
            )
        except Exception as e:
            await self._fail(context, event_queue, task.id, trace_id, f"Edit call error: {e}")
            return
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        if result.get("status_code") != 200 or not result.get("results"):
            await self._fail(context, event_queue, task.id, trace_id, f"{result.get('code')} - {result.get('message')}")
            return

        await self._emit_progress(event_queue, context, task.id, trace_id, "model_done", 60,
                                  {"count": len(result["results"]), "latency_ms": latency_ms})

        # download & per-file progress
        saved_files: List[Dict[str, str]] = []
        urls: List[str] = result.get("results", [])
        if params.get("save", True) and urls:
            save_dir = Path(params.get("save_dir", "outputs/edits"))
            overwrite = bool(params.get("overwrite", False))
            _ensure_dir(save_dir)

            await self._emit_progress(event_queue, context, task.id, trace_id, "downloading", 65,
                                      {"dir": str(save_dir), "total": len(urls)})

            current = 0
            lock = anyio.Lock()

            async def _download_one(i: int, url: str):
                nonlocal current, saved_files
                name = _sanitize_name(Path(url).stem or f"edit_{i}")
                target = save_dir / f"{name}_{i}"
                try:
                    p = await anyio.to_thread.run_sync(self.core._blocking_download, url, target, overwrite)
                    rec = {"url": url, "path": str(p)}
                except Exception as e:
                    rec = {"url": url, "path": "", "error": str(e)}
                async with lock:
                    saved_files.append(rec); current += 1
                    percent = 70 + int(25 * current / max(1, len(urls)))  # 70~95
                    await self._emit_progress(event_queue, context, task.id, trace_id, "saved_file", percent,
                                              {**rec, "index": i}, artifact_name="image_edit_saved")

            async with anyio.create_task_group() as tg:
                for i, u in enumerate(urls):
                    tg.start_soon(_download_one, i, u)
        else:
            await self._emit_progress(event_queue, context, task.id, trace_id, "skip_download", 90,
                                      {"reason": "save_disabled_or_no_results"})

        # final artifact
        human = (
            "✅ 图像编辑完成\n"
            f"模型: {result.get('model')}\n"
            f"参数: {json.dumps(result.get('request'), ensure_ascii=False)}\n"
            "输出链接：\n" + "\n".join(f"- {u}" for u in result["results"])
        )
        payload = {"traceId": trace_id, **result, "saved_files": saved_files}
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="image_edit_result", text=human + "\n\n<result.json>\n" + json.dumps(payload, ensure_ascii=False, indent=2)),
            context_id=context.context_id, task_id=task.id,
        ))

        await self._emit_progress(event_queue, context, task.id, trace_id, "succeeded", 100)

        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task.id,
            status=TaskStatus(state=TaskState.completed), final=True,
        ))

    async def _fail(self, context: RequestContext, event_queue: EventQueue, task_id: str, trace_id: str, msg: str):
        logger.error("[trace:%s] FAILED | %s", trace_id, msg)
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="image_edit_error", text=f"[trace:{trace_id}] ❌ {msg}"),
            context_id=context.context_id, task_id=task_id,
        ))
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="image_edit_status", text=_status_json("failed", trace_id, {"error": msg})),
            context_id=context.context_id, task_id=task_id,
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task_id,
            status=TaskStatus(state=TaskState.completed), final=True,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = getattr(context, "task_id", None) or "unknown"
        canceled_state = getattr(TaskState, "canceled", TaskState.completed)
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="image_edit_status", text=_status_json("failed", uuid.uuid4().hex[:8], {"canceled": True})),
            context_id=context.context_id, task_id=task_id,
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id, task_id=task_id,
            status=TaskStatus(state=canceled_state), final=True,
        ))
