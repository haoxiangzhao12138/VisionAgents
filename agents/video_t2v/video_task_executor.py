# video_task_executor.py
import os, re, json, time, uuid, anyio, logging, mimetypes, requests
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
    DEFAULT_VIDEO_SIZE,
)

logger = logging.getLogger("video_task_executor")

# -----------------------
# Small helpers
# -----------------------
def _status_json(state: str, trace_id: str, payload: Dict[str, Any] | None = None) -> str:
    return json.dumps({
        "type": "video_status",
        "state": state,                # started | saved_file | succeeded | failed
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

def _infer_video_ext(headers: Dict[str, str], default=".mp4") -> str:
    ct = headers.get("Content-Type") or headers.get("content-type") or ""
    ext = mimetypes.guess_extension(ct.split(";")[0].strip()) if ct else None
    if not ext:
        if "video/mp4" in ct: ext = ".mp4"
        elif "video/webm" in ct: ext = ".webm"
        elif "video/quicktime" in ct or "video/mov" in ct: ext = ".mov"
    return ext or default

def _filename_from_url(url: str, default_ext=".mp4") -> str:
    base = url.split("?")[0].rstrip("/")
    name = base.split("/")[-1] if "/" in base else base
    name = _sanitize_filename(name)
    root, ext = os.path.splitext(name)
    if not ext: name = f"{name}{default_ext}"
    return name

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------
# Param extraction
# -----------------------
def _extract_user_params(context: RequestContext) -> Dict[str, Any]:
    """
    支持：
      1) 纯文本 -> 作为 prompt
      2) 文本里包含 JSON -> 解析 JSON 并合并

    可用字段（DashScope T2V 对齐）：
      prompt (str) 必填
      audio_url (str) 可选
      size (str) 默认 '832*480'
      duration (int) 默认 6
      negative_prompt (str) 可选
      prompt_extend (bool) 默认 True
      watermark (bool) 默认 False
      seed (int) 可选
      model (str) 默认 'wan2.5-t2v-preview'

    保存控制：
      save (bool, default True), save_dir (str, default 'videos'), overwrite (bool, default False)
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
    params.setdefault("size", os.getenv("DASHSCOPE_T2V_SIZE", DEFAULT_VIDEO_SIZE))
    params.setdefault("duration", int(os.getenv("DASHSCOPE_T2V_DURATION", str(DEFAULT_VIDEO_DURATION))))
    params.setdefault("prompt_extend", True)
    params.setdefault("watermark", False)
    params.setdefault("model", os.getenv("DASHSCOPE_T2V_MODEL", "wan2.5-t2v-preview"))

    params.setdefault("save", DEFAULT_SAVE_ARTIFACTS)
    params.setdefault("save_dir", os.getenv("VIDEO_SAVE_DIR", "videos"))
    params.setdefault("overwrite", False)

    dashscope.base_http_api_url = os.getenv("DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL)
    return params

# -----------------------
# Core Video Agent
# -----------------------
class VideoGenAgent:
    """调用 DashScope 文生视频并（可选）下载保存。"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")

    def _blocking_call(self, call_kwargs: Dict[str, Any]):
        return VideoSynthesis.call(**call_kwargs)

    def _blocking_download(self, url: str, target: Path, overwrite: bool) -> Path:
        if target.exists() and not overwrite:
            return target
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        if target.suffix == "":
            target = target.with_suffix(_infer_video_ext(r.headers, default=".mp4"))
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk: f.write(chunk)
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
            size=params.get("size", "832*480"),
            duration=int(params.get("duration", 6)),
            prompt_extend=bool(params.get("prompt_extend", True)),
            watermark=bool(params.get("watermark", False)),
        )
        if params.get("seed") is not None:
            call_kwargs["seed"] = int(params["seed"])
        if params.get("audio_url"):
            call_kwargs["audio_url"] = params["audio_url"]

        rsp = await anyio.to_thread.run_sync(self._blocking_call, call_kwargs)

        if rsp.status_code != HTTPStatus.OK:
            return {
                "status_code": int(rsp.status_code),
                "code": getattr(rsp, "code", "DashScopeError"),
                "message": getattr(rsp, "message", "Unknown error from DashScope"),
            }

        video_url = getattr(getattr(rsp, "output", None), "video_url", None)
        results = [{"url": video_url, "index": 0}] if video_url else []
        out: Dict[str, Any] = {
            "status_code": 200,
            "model": call_kwargs["model"],
            "request_params": {k: v for k, v in call_kwargs.items() if k != "api_key"},
            "results": results,
        }

        # 可选保存
        if params.get("save", True) and video_url:
            save_dir = Path(params.get("save_dir", "videos"))
            _ensure_dir(save_dir)
            name = _filename_from_url(video_url, default_ext=".mp4")
            base, ext = os.path.splitext(name)
            candidate = save_dir / f"{base}{ext}"
            try:
                p = await anyio.to_thread.run_sync(
                    self._blocking_download, video_url, candidate, bool(params.get("overwrite", False))
                )
                out["saved_files"] = [{"url": video_url, "path": str(p)}]
            except Exception as e:
                out["saved_files"] = [{"url": video_url, "path": "", "error": str(e)}]

        return out

# -----------------------
# Task-based Executor
# -----------------------
class VideoTaskAgentExecutor(AgentExecutor):
    """
    文生视频执行器（Holos Task API）：
      1) new_task
      2) TaskStatus: working
      3) 若保存：保存成功立刻发中间 TaskArtifact（文本+JSON状态）
      4) 最终 TaskArtifact（汇总）
      5) TaskStatus: completed(final=True)
    """

    def __init__(self):
        # ❌ 移除硬编码 key，改成环境变量；缺失会在 generate 里返回清晰错误
        self.agent = VideoGenAgent()

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
            final=False,
        ))

        params = _extract_user_params(context)
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="video_status",
                text=_status_json("started", trace_id, {
                    "model": params.get("model"),
                    "size": params.get("size"),
                    "duration": params.get("duration")
                })
            ),
            context_id=context.context_id,
            task_id=a2a_task.id,
        ))

        # 3) 生成
        result = await self.agent.generate(params)

        # 保存成功即时提示
        for f in result.get("saved_files", []) or []:
            if f.get("path"):
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(
                        name="video_status",
                        text=_status_json("saved_file", trace_id, {"path": f["path"], "url": f["url"]})
                    ),
                    context_id=context.context_id,
                    task_id=a2a_task.id,
                ))

        # 4) 最终 artifact
        if result.get("status_code") == 200 and result.get("results"):
            video_url = result["results"][0]["url"] if result["results"] else ""
            saved_lines: List[str] = []
            for f in result.get("saved_files", []) or []:
                if f.get("path"):
                    saved_lines.append(f"- {f['path']}  ← {f['url']}")
                else:
                    saved_lines.append(f"- [下载失败] {f['url']} ({f.get('error','')})")
            saved_part = "\n本地保存：\n" + "\n".join(saved_lines) if saved_lines else ""

            payload = {
                "traceId": trace_id,
                "model": result.get("model"),
                "request": result.get("request_params", {}),
                "videoUrl": video_url,
                "saved_files": result.get("saved_files", []),
            }
            human = (
                "🎬 视频生成成功\n"
                f"模型: {payload['model']}\n"
                f"参数: {json.dumps(payload['request'], ensure_ascii=False)}\n"
                f"视频链接：\n- {video_url}{saved_part}"
            )
            final_json = _status_json("succeeded", trace_id, payload)

            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="video_result", text=human + "\n\n<result.json>\n" + json.dumps(payload, ensure_ascii=False, indent=2)),
                context_id=context.context_id, task_id=a2a_task.id,
            ))
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="video_status", text=final_json),
                context_id=context.context_id, task_id=a2a_task.id,
            ))
        else:
            err_payload = {
                "traceId": trace_id,
                "status_code": result.get("status_code"),
                "code": result.get("code"),
                "message": result.get("message"),
            }
            human = (
                "❌ 视频生成失败\n"
                f"status_code: {err_payload['status_code']}\n"
                f"code: {err_payload['code']}\n"
                f"message: {err_payload['message']}"
            )
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="video_error", text=human),
                context_id=context.context_id, task_id=a2a_task.id,
            ))
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                artifact=new_text_artifact(name="video_status", text=_status_json("failed", trace_id, err_payload)),
                context_id=context.context_id, task_id=a2a_task.id,
            ))

        # 5) completed
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            context_id=context.context_id,
            task_id=a2a_task.id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = getattr(context, "task_id", None) or "unknown"
        canceled_state = getattr(TaskState, "canceled", TaskState.completed)

        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            artifact=new_text_artifact(
                name="video_status",
                text="🛑 任务已取消"
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
