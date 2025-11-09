# image_task_executor.py
import os, re, json, time, uuid, anyio, logging, mimetypes, requests
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional

import dashscope
from dashscope import ImageSynthesis

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_text_artifact
from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus, TaskArtifactUpdateEvent

logger = logging.getLogger("image_task_executor")

# -----------------------
# Small helpers
# -----------------------
def _status_json(state: str, trace_id: str, payload: Dict[str, Any] | None = None) -> str:
    return json.dumps({
        "type": "image_status",
        "state": state,                # started | generating | downloading | saved_file | succeeded | failed
        "traceId": trace_id,
        "timestamp": time.time(),
        "data": payload or {},
    }, ensure_ascii=False)

def _safe_get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _to_dict(obj: Any) -> Any:
    try:
        if hasattr(obj, "model_dump"): return obj.model_dump()
        if hasattr(obj, "dict"):       return obj.dict()
    except Exception:
        pass
    if hasattr(obj, "__dict__"):
        out = {}
        for k in dir(obj):
            if not k.startswith("_"):
                try: out[k] = getattr(obj, k)
                except Exception: pass
        return out
    return obj

def _get_message_from_context(context: RequestContext) -> Any:
    for path in [
        ("message",),
        ("params", "message"),
        ("request", "message"),
        ("input", "message"),
        ("input_message",),
    ]:
        cur = context
        ok = True
        for key in path:
            cur = _safe_get(cur, key, None)
            if cur is None: ok = False; break
        if ok and cur is not None: return cur
    return getattr(context, "params", None)

def _deep_collect_texts(node: Any, bag: List[str]) -> None:
    if node is None: return
    if isinstance(node, str):
        s = node.strip()
        if s: bag.append(s)
        return
    if isinstance(node, (list, tuple)):
        for it in node: _deep_collect_texts(it, bag)
        return
    obj = _to_dict(node)
    if isinstance(obj, dict):
        parts = obj.get("parts") or obj.get("content") or obj.get("contents")
        if isinstance(parts, (list, tuple)):
            for p in parts:
                txt = _safe_get(p, "text", None)
                if isinstance(txt, str) and txt.strip():
                    bag.append(txt.strip())
        txt = obj.get("text")
        if isinstance(txt, str) and txt.strip(): bag.append(txt.strip())
        for v in obj.values(): _deep_collect_texts(v, bag)

def _maybe_parse_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    start, end = s.find("{"), s.rfind("}")
    if 0 <= start < end:
        maybe_json = s[start:end+1]
        try:
            parsed = json.loads(maybe_json)
            if isinstance(parsed, dict): return parsed
        except Exception:
            return None
    return None

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or uuid.uuid4().hex

def _infer_ext_from_headers(headers: Dict[str, str]) -> str:
    ct = headers.get("Content-Type") or headers.get("content-type") or ""
    ext = mimetypes.guess_extension(ct.split(";")[0].strip()) if ct else None
    if not ext:
        if "image/png" in ct: ext = ".png"
        elif "image/jpeg" in ct or "image/jpg" in ct: ext = ".jpg"
        elif "image/webp" in ct: ext = ".webp"
        elif "image/gif" in ct: ext = ".gif"
    return ext or ".png"

def _filename_from_url(url: str) -> str:
    base = url.split("?")[0].rstrip("/")
    name = base.split("/")[-1] if "/" in base else base
    name = _sanitize_filename(name)
    if not os.path.splitext(name)[1]:
        name = f"{name}.png"
    return name

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------
# Param extraction
# -----------------------
def _extract_user_params(context: RequestContext) -> Dict[str, Any]:
    """
    支持文本/JSON，默认对齐 DashScope T2I：
      prompt, negative_prompt, n, size, seed, model, prompt_extend, watermark
    保存控制：
      save (bool, default True), save_dir (str, default 'outputs'), overwrite (bool, default False)
    """
    msg = _get_message_from_context(context)
    texts: List[str] = []
    _deep_collect_texts(msg, texts)
    first_text = texts[0] if texts else ""

    params: Dict[str, Any] = {}
    if first_text:
        parsed = _maybe_parse_json_from_text(first_text)
        if parsed: params.update(parsed)
        if not params.get("prompt"): params["prompt"] = first_text

    params.setdefault("negative_prompt", "")
    params.setdefault("n", 1)
    params.setdefault("size", os.getenv("DASHSCOPE_IMAGE_SIZE", "1024*1024"))
    params.setdefault("prompt_extend", True)
    params.setdefault("watermark", False)
    params.setdefault("model", os.getenv("DASHSCOPE_MODEL", "wan2.5-t2i-preview"))

    params.setdefault("save", True)
    params.setdefault("save_dir", os.getenv("IMAGE_SAVE_DIR", "outputs"))
    params.setdefault("overwrite", False)

    dashscope.base_http_api_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v1",  # 海外可改为 https://dashscope-intl.aliyuncs.com/api/v1
    )
    return params

# -----------------------
# Core Image Agent
# -----------------------
class ImageGenAgent:
    """调用 DashScope 文生图并（可选）下载保存。"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

    def _blocking_call(self, call_kwargs: Dict[str, Any]):
        # 提高稳定性：DashScope SDK 内部会处理重试；必要时可加超时控制
        return ImageSynthesis.call(**call_kwargs)

    def _blocking_download(self, url: str, target: Path, overwrite: bool) -> Path:
        if target.exists() and not overwrite:
            return target
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        if target.suffix == "":
            target = target.with_suffix(_infer_ext_from_headers(r.headers))
        with open(target, "wb") as f:
            f.write(r.content)
        return target

    async def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return {"status_code": 400, "code": "MissingAPIKey", "message": "DASHSCOPE_API_KEY is not set."}

        prompt = params.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return {"status_code": 400, "code": "MissingPrompt", "message": "Parameter 'prompt' is required."}

        call_kwargs: Dict[str, Any] = dict(
            api_key=self.api_key,
            model=params.get("model"),
            prompt=prompt,
            negative_prompt=params.get("negative_prompt", ""),
            n=int(params.get("n", 1)),
            size=params.get("size", "1024*1024"),
            prompt_extend=bool(params.get("prompt_extend", True)),
            watermark=bool(params.get("watermark", False)),
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

        results = [{"url": r.url, "index": i} for i, r in enumerate(getattr(rsp.output, "results", []) or [])]
        out: Dict[str, Any] = {
            "status_code": 200,
            "model": call_kwargs["model"],
            "request_params": {k: v for k, v in call_kwargs.items() if k != "api_key"},
            "results": results,
        }

        # 可选保存
        if params.get("save", True) and results:
            save_dir = Path(params.get("save_dir", "outputs"))
            _ensure_dir(save_dir)
            saved_files: List[Dict[str, str]] = []

            async def _save_one(item: Dict[str, Any]) -> Dict[str, str]:
                url = item["url"]
                name = _filename_from_url(url)
                base, ext = os.path.splitext(name)
                candidate = save_dir / f"{base}_{item['index']}{ext}"
                try:
                    p = await anyio.to_thread.run_sync(
                        self._blocking_download, url, candidate, bool(params.get("overwrite", False))
                    )
                    return {"url": url, "path": str(p)}
                except Exception as e:
                    return {"url": url, "path": "", "error": str(e)}

            results_holder: List[Dict[str, str]] = []

            async with anyio.create_task_group() as tg:
                async def run_and_collect(it):
                    res = await _save_one(it)
                    results_holder.append(res)
                for it in results:
                    tg.start_soon(run_and_collect, it)

            out["saved_files"] = results_holder

        return out

# -----------------------
# Task-based Executor
# -----------------------
class ImageTaskAgentExecutor(AgentExecutor):
    """
    文生图执行器（Holos Task API）：
      - 事件更“瘦身”：只发关键状态 + 紧凑结果 JSON
      - 事件序列：started -> generating -> (downloading/saved_file)* -> image_result -> succeeded
    """

    def __init__(self):
        # ❌ 之前是硬编码 API Key；现在改为仅读环境变量
        self.agent = ImageGenAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        trace_id = uuid.uuid4().hex[:8]

        # 1) 创建任务
        a2a_task = new_task(context.message)
        a2a_task.id = context.task_id
        await event_queue.enqueue_event(a2a_task)

        # 2) working
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id,
            task_id=a2a_task.id,
            status=TaskStatus(state=TaskState.working),
            final=False
        ))

        params = _extract_user_params(context)
        # started
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="image_status",
                text=_status_json("started", trace_id, {
                    "model": params.get("model"),
                    "n": params.get("n"),
                    "size": params.get("size")
                })
            ),
            context_id=context.context_id,
            task_id=a2a_task.id,
        ))
        # generating
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="image_status",
                text=_status_json("generating", trace_id, {"prompt_preview": (params.get("prompt") or "")[:60]})
            ),
            context_id=context.context_id,
            task_id=a2a_task.id,
        ))

        # 3) 生成
        result = await self.agent.generate(params)

        # 错误
        if result.get("status_code") != 200 or not result.get("results"):
            err_payload = {
                "traceId": trace_id,
                "status_code": result.get("status_code"),
                "code": result.get("code"),
                "message": result.get("message"),
            }
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="image_error", text=json.dumps(err_payload, ensure_ascii=False)),
                context_id=context.context_id, task_id=a2a_task.id,
            ))
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="image_status", text=_status_json("failed", trace_id, err_payload)),
                context_id=context.context_id, task_id=a2a_task.id,
            ))
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                context_id=context.context_id,
                task_id=a2a_task.id,
                status=TaskStatus(state=TaskState.completed),
                final=True,
            ))
            return

        # 4) 下载阶段进展
        if params.get("save", True):
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(
                    name="image_status",
                    text=_status_json("downloading", trace_id, {"count": len(result.get("results", []))})
                ),
                context_id=context.context_id, task_id=a2a_task.id,
            ))
            # saved_file 事件
            for f in result.get("saved_files", []) or []:
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(
                        name="image_saved",
                        text=_status_json("saved_file", trace_id, {k: v for k, v in f.items() if k in ("path", "url", "error")})
                    ),
                    context_id=context.context_id,
                    task_id=a2a_task.id,
                ))

        # 5) 紧凑结果产物
        compact_results: List[Dict[str, str]] = []
        saved_map: Dict[str, str] = {f.get("url"): f.get("path") for f in (result.get("saved_files") or []) if f.get("path")}
        for r in result.get("results", []):
            u = r.get("url")
            compact_results.append({"url": u, "path": saved_map.get(u, "")})

        compact_payload = {
            "traceId": trace_id,
            "model": result.get("model"),
            "prompt": (result.get("request_params") or {}).get("prompt", ""),
            "results": compact_results,
            "count": len(compact_results),
        }
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="image_result", text=json.dumps(compact_payload, ensure_ascii=False)),
            context_id=context.context_id, task_id=a2a_task.id,
        ))

        # 6) 成功状态
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(name="image_status", text=_status_json("succeeded", trace_id, {"count": len(compact_results)})),
            context_id=context.context_id, task_id=a2a_task.id,
        ))

        # 7) completed
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id,
            task_id=a2a_task.id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        发送“已取消”的任务状态；如果 TaskState 没有 canceled，就回退成 completed。
        """
        task_id = getattr(context, "task_id", None) or "unknown"
        canceled_state = getattr(TaskState, "canceled", TaskState.completed)

        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="image_status",
                text=_status_json("failed", uuid.uuid4().hex[:8], {"canceled": True})
            ),
            context_id=context.context_id,
            task_id=task_id,
        ))
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id,
            task_id=task_id,
            status=TaskStatus(state=canceled_state),
            final=True,
        ))
