"""Convenience launcher that boots all services for local development."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

SERVICES: Sequence[tuple[str, str]] = (
    ("planning", "planning.server_planning"),
    ("router", "router.server_router"),
    ("prompt", "agents.prompt_enhance.server_prompt"),
    ("image_t2i", "agents.image_t2i.server_image"),
    ("image_edit", "agents.image_edit.server_image_edit"),
    ("video_t2v", "agents.video_t2v.server_video"),
    ("video_i2v", "agents.video_i2v.server_video_i2v"),
)


@dataclass
class ManagedProcess:
    name: str
    module: str
    process: asyncio.subprocess.Process


async def _stream_output(name: str, stream: asyncio.StreamReader, prefix: str) -> None:
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            sys.stdout.write(f"[{prefix}:{name}] {line.decode().rstrip()}\n")
            sys.stdout.flush()
    except asyncio.CancelledError:
        pass


async def _start_service(name: str, module: str, env: Dict[str, str]) -> ManagedProcess:
    python = sys.executable
    process = await asyncio.create_subprocess_exec(
        python,
        "-m",
        module,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    return ManagedProcess(name=name, module=module, process=process)


async def _run_services(targets: Iterable[tuple[str, str]], stop_event: asyncio.Event) -> None:
    env = os.environ.copy()
    managed: List[ManagedProcess] = []

    for name, module in targets:
        proc = await _start_service(name, module, env)
        managed.append(proc)
        print(f"[start-all] spawned {name} ({module}) pid={proc.process.pid}")

    tasks = []
    for proc in managed:
        assert proc.process.stdout and proc.process.stderr
        tasks.append(asyncio.create_task(_stream_output(proc.name, proc.process.stdout, "stdout")))
        tasks.append(asyncio.create_task(_stream_output(proc.name, proc.process.stderr, "stderr")))

    try:
        waiters = [asyncio.create_task(p.process.wait()) for p in managed]
        _, pending = await asyncio.wait(
            waiters + [asyncio.create_task(stop_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_event.is_set():
            print("[start-all] stop requested, terminating child processes...")
            for proc in managed:
                if proc.process.returncode is None:
                    proc.process.terminate()
            await asyncio.gather(*[p.process.wait() for p in managed], return_exceptions=True)
        else:
            for fut in pending:
                fut.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start all VisionAgents services with a single command.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[name for name, _ in SERVICES],
        help="Optionally specify a subset of services to launch.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    targets = SERVICES
    if args.only:
        selected = {name: module for name, module in SERVICES}
        missing = [name for name in args.only if name not in selected]
        if missing:
            raise SystemExit(f"Unknown services requested: {', '.join(missing)}")
        targets = [(name, selected[name]) for name in args.only]

    loop = asyncio.get_event_loop()

    stop_event = asyncio.Event()

    def _signal_handler(signum, _frame):  # type: ignore[override]
        print(f"[start-all] received signal {signum}, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler, sig, None)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: asyncio.ensure_future(stop_event.set()))

    async def runner():
        await _run_services(targets, stop_event)

    try:
        loop.run_until_complete(runner())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
