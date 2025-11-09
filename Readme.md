# VisionAgents

A collection of Holos-compatible agent services for visual content planning and generation.  Each service exposes an HTTP API
using the [`a2a`](https://github.com/modelcontextprotocol/a2a) framework and can be started independently.

## Services

| Module | Command | Description |
| --- | --- | --- |
| `planning/server_planning.py` | `python -m planning.server_planning` | Generates Holos plans with the Yunstorm Azure OpenAI endpoint. |
| `router/server_router.py` | `python -m router.server_router` | Resolves Holos plans and dispatches sub tasks to downstream agents. |
| `agents/prompt_enhance/server_prompt.py` | `python -m agents.prompt_enhance.server_prompt` | Enhances user prompts before generation. |
| `agents/image_t2i/server_image.py` | `python -m agents.image_t2i.server_image` | DashScope text-to-image executor. |
| `agents/image_edit/server_image_edit.py` | `python -m agents.image_edit.server_image_edit` | DashScope image edit executor. |
| `agents/video_t2v/server_video.py` | `python -m agents.video_t2v.server_video` | DashScope text-to-video executor. |
| `agents/video_i2v/server_video_i2v.py` | `python -m agents.video_i2v.server_video_i2v` | DashScope image-to-video executor. |

All services share the configuration in `shared/config.py`.  Environment variables can be used to override ports, public URLs, and
API keys without modifying the code.

## Quick start

1. Install dependencies (simplified example):
   ```bash
   pip install -r requirements.txt  # include a2a, holos_sdk, openai, dashscope, httpx
   ```
2. Export credentials (replace with real keys):
   ```bash
   export YUNSTORM_API_KEY="..."
   export DASHSCOPE_API_KEY="..."
   ```
3. Start the planning agent:
   ```bash
   python -m planning.server_planning
   ```
4. Start the router and any downstream task agents you need.  The router uses the default localhost ports from
   `shared/config.py`, so running each command in a separate terminal is enough for a local development setup.

   Alternatively you can launch every service with a single command:

   ```bash
   python scripts/start_all.py
   ```

   Use `--only planning router` to boot a subset of services.

5. Connect Holos' official front-end to the planning endpoint exposed by `planning.server_planning`.  The
   planning executor attaches Holos tracing metadata to the downstream calls so that the built-in front-end
   can display task progress and intermediate artifacts without additional glue code.

Health endpoints are available at `http://localhost:<port>/health` for all services.
