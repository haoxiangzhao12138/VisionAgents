"""Utilities shared across task executors."""

import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import requests

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
    """Download ``url`` to ``target`` asynchronously using a worker thread."""

    if target.exists() and not overwrite:
        return target

    def _blocking() -> Path:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        if target.suffix:
            target_path = target
        else:
            ext = guess_ext_from_headers(dict(response.headers), fallback_ext)
            target_path = target.with_suffix(ext)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    fh.write(chunk)
        return target_path

    return await anyio.to_thread.run_sync(_blocking)
