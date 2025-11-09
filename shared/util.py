import json, time, re, uuid, mimetypes, requests, anyio
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- artifacts helpers ----
def status_json(_type: str, state: str, trace_id: str, payload: Dict[str, Any] | None = None) -> str:
    return json.dumps({"type": _type, "state": state, "traceId": trace_id, "timestamp": time.time(), "data": payload or {}}, ensure_ascii=False)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or uuid.uuid4().hex

def guess_ext_from_headers(headers: Dict[str, str], fallback: str) -> str:
    ct = headers.get("Content-Type") or headers.get("content-type") or ""
    ext = mimetypes.guess_extension(ct.split(";")[0].strip()) if ct else None
    return ext or fallback

async def download_file(url: str, target: Path, overwrite: bool, fallback_ext: str) -> Path:
    if target.exists() and not overwrite:
        return target
    def _blocking():
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        if target.suffix == "":
            from .util import guess_ext_from_headers
            target_with = target.with_suffix(guess_ext_from_headers(r.headers, fallback_ext))
        else:
            target_with = target
        target_with.parent.mkdir(parents=True, exist_ok=True)
        with open(target_with, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*256):
                if chunk: f.write(chunk)
        return target_with
    return await anyio.to_thread.run_sync(_blocking)