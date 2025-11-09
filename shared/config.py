"""Centralised configuration helpers used across all services."""

from __future__ import annotations

import os
from typing import Final


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment with a sane fallback."""

    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ── Shared network configuration ------------------------------------------------
HOST: Final[str] = os.getenv("HOST", "0.0.0.0")
TRACING_API_BASE: Final[str] = os.getenv("HOLOS_TRACER_BASE_URL", "http://localhost:8000/api/v1")


def _public_url(port: int, env_name: str) -> str:
    return os.getenv(env_name, f"http://localhost:{port}/")


ROUTER_PORT: Final[int] = _env_int("ROUTER_PORT", 10102)
ROUTER_PUBLIC_URL: Final[str] = _public_url(ROUTER_PORT, "ROUTER_PUBLIC_URL")

PLANNING_PORT: Final[int] = _env_int("PLANNING_PORT", 10101)
PLANNING_PUBLIC_URL: Final[str] = _public_url(PLANNING_PORT, "PLANNING_PUBLIC_URL")

IMAGE_T2I_PORT: Final[int] = _env_int("IMAGE_T2I_PORT", 10110)
IMAGE_T2I_PUBLIC_URL: Final[str] = _public_url(IMAGE_T2I_PORT, "IMAGE_T2I_PUBLIC_URL")

VIDEO_T2V_PORT: Final[int] = _env_int("VIDEO_T2V_PORT", 10111)
VIDEO_T2V_PUBLIC_URL: Final[str] = _public_url(VIDEO_T2V_PORT, "VIDEO_T2V_PUBLIC_URL")

VIDEO_I2V_PORT: Final[int] = _env_int("VIDEO_I2V_PORT", 10112)
VIDEO_I2V_PUBLIC_URL: Final[str] = _public_url(VIDEO_I2V_PORT, "VIDEO_I2V_PUBLIC_URL")

IMAGE_EDIT_PORT: Final[int] = _env_int("IMAGE_EDIT_PORT", 10113)
IMAGE_EDIT_PUBLIC_URL: Final[str] = _public_url(IMAGE_EDIT_PORT, "IMAGE_EDIT_PUBLIC_URL")

PROMPT_PORT: Final[int] = _env_int("PROMPT_PORT", 10130)
PROMPT_PUBLIC_URL: Final[str] = _public_url(PROMPT_PORT, "PROMPT_PUBLIC_URL")


# ── Router upstream URLs --------------------------------------------------------
# Router discovers downstream agents via their public URLs. Keep the defaults
# aligned with the ports above so the services work out-of-the-box locally.
IMAGE_T2I_URL: Final[str] = os.getenv("IMAGE_T2I_URL", IMAGE_T2I_PUBLIC_URL)
VIDEO_T2V_URL: Final[str] = os.getenv("VIDEO_T2V_URL", VIDEO_T2V_PUBLIC_URL)
VIDEO_I2V_URL: Final[str] = os.getenv("VIDEO_I2V_URL", VIDEO_I2V_PUBLIC_URL)
IMAGE_EDIT_URL: Final[str] = os.getenv("IMAGE_EDIT_URL", IMAGE_EDIT_PUBLIC_URL)
PROMPT_URL: Final[str] = os.getenv("PROMPT_URL", PROMPT_PUBLIC_URL)


# ── LLM / third-party integrations ---------------------------------------------
YUNSTORM_ENDPOINT: Final[str] = os.getenv("YUNSTORM_ENDPOINT", "https://gpt.yunstorm.com/")
YUNSTORM_API_KEY: Final[str] = os.getenv("YUNSTORM_API_KEY", "")
YUNSTORM_API_VERSION: Final[str] = os.getenv("YUNSTORM_API_VERSION", "2025-04-01-preview")
YUNSTORM_MODEL: Final[str] = os.getenv("YUNSTORM_MODEL", "gpt-4.1")

DASHSCOPE_API_KEY: Final[str] = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL: Final[str] = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")


# ── Default generation parameters ----------------------------------------------
DEFAULT_IMAGE_SIZE: Final[str] = os.getenv("DEFAULT_IMAGE_SIZE", "1024*1024")
DEFAULT_VIDEO_SIZE: Final[str] = os.getenv("DEFAULT_VIDEO_SIZE", "832*480")
DEFAULT_VIDEO_DURATION: Final[int] = _env_int("DEFAULT_VIDEO_DURATION", 8)
DEFAULT_SAVE_ARTIFACTS: Final[bool] = os.getenv("DEFAULT_SAVE_ARTIFACTS", "0") == "1"
