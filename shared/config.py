# shared/config.py
import os

# ── A2A/Holos trace 后端
TRACING_API_BASE = os.getenv("HOLOS_TRACER_BASE_URL", 'https://nat-notebook-inspire.sii.edu.cn/ws-677a4cfa-ac67-494c-b481-e9147a3487a2/project-034ec99f-f57a-4c71-9f7e-1654d2c430c8/user-dc5518ff-a2c5-45ed-a36f-3931cb726d94/vscode/9b87a33b-9112-4cf2-90b9-de6b42e6072a/c8f23942-50b9-4630-ae9a-ce48e66b7281/proxy/8000/api/v1')

# ── 服务对外 URL（供 Router 用 A2ACardResolver 加载）
IMAGE_T2I_URL = os.getenv("IMAGE_T2I_URL",  "http://localhost:10110/")
VIDEO_T2V_URL = os.getenv("VIDEO_T2V_URL",  "http://localhost:10111/")
VIDEO_I2V_URL = os.getenv("VIDEO_I2V_URL",  "http://localhost:10112/")
IMAGE_EDIT_URL = os.getenv("IMAGE_EDIT_URL","http://localhost:10113/")
PROMPT_URL    = os.getenv("PROMPT_URL",     "http://localhost:10130/")

# ── 云风网关 / DashScope
YUNSTORM_ENDPOINT = os.getenv("YUNSTORM_ENDPOINT", "https://gpt.yunstorm.com/")
YUNSTORM_API_KEY  = os.getenv("YUNSTORM_API_KEY", "c1660c7c06c32f4a48c5bac00e5852a5")
YUNSTORM_API_VERSION = os.getenv("YUNSTORM_API_VERSION", "2025-04-01-preview")
YUNSTORM_MODEL = os.getenv("YUNSTORM_MODEL", "gpt-4.1")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-a4ea7d22270b4ae28c3a5ed948b05897")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")

# ── 默认规格（URL-only 场景：save 默认 False）
DEFAULT_IMAGE_SIZE = os.getenv("DEFAULT_IMAGE_SIZE", "1024*1024")
DEFAULT_VIDEO_SIZE = os.getenv("DEFAULT_VIDEO_SIZE", "832*480")
DEFAULT_VIDEO_DURATION = int(os.getenv("DEFAULT_VIDEO_DURATION", "8"))
DEFAULT_SAVE_ARTIFACTS = os.getenv("DEFAULT_SAVE_ARTIFACTS", "0") == "1"  # 默认不保存
