"""Subprocess-based Docker sandbox for running an episode in isolation.

Each episode gets its own container. The repo + policy artifacts are mounted
read-write under ``/work`` (mutated freely; the sandbox is disposed after);
CI artifacts are read back via a results mount.

Why subprocess instead of the ``docker`` SDK: subprocess gives us a portable
fallback to ``podman`` (drop-in CLI compat), works in CI without the SDK
installed, and produces logs that are easy to commit alongside the
trajectory for the paper's reproducibility appendix.

The ``run_in_sandbox`` coroutine is a *thin wrapper around the CLI*.
Anything stronger (pause/resume, network policy) is added when needed.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    """Container exit info + paths to the artifact files emitted by CI."""

    exit_code: int
    stdout: str
    stderr: str
    artifact_dir: Path
    duration_s: float


@asynccontextmanager
async def docker_sandbox(
    image: str,
    repo_path: Path,
    *,
    docker_bin: str = "docker",
) -> AsyncIterator[Path]:
    """Yield a fresh tmp dir to which a container will write artifacts.

    The caller invokes :func:`run_in_sandbox` inside this block. After the
    block exits, all temp state is removed.
    """
    tmp = Path(tempfile.mkdtemp(prefix="carl_episode_"))
    try:
        # Pre-flight: image must be available; pull if not.
        await _docker_pull(docker_bin, image)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_in_sandbox(
    image: str,
    repo_path: Path,
    artifact_dir: Path,
    cmd: list[str],
    *,
    timeout_s: int,
    docker_bin: str = "docker",
    extra_env: dict[str, str] | None = None,
) -> SandboxResult:
    """Run ``cmd`` in a fresh container; mount ``repo_path`` and ``artifact_dir``.

    The container's working directory is ``/work`` (the mounted repo). The
    CI is expected to write its artifacts (pytest JSON, coverage XML, …)
    into ``/artifacts``, which the sandbox tmp directory mounts to.
    """
    import time

    docker_args = [
        docker_bin, "run", "--rm",
        "--mount", f"type=bind,src={repo_path},dst=/work",
        "--mount", f"type=bind,src={artifact_dir},dst=/artifacts",
        "--workdir", "/work",
    ]
    for k, v in (extra_env or {}).items():
        docker_args += ["-e", f"{k}={v}"]
    docker_args += [image, *cmd]

    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *docker_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        exit_code = int(proc.returncode or 0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        stdout, stderr = b"", b""
        exit_code = 124  # GNU timeout convention
    duration = time.monotonic() - start

    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        artifact_dir=artifact_dir,
        duration_s=duration,
    )


async def _docker_pull(docker_bin: str, image: str) -> None:
    """``docker pull`` the image if it isn't present locally; cheap when cached."""
    proc = await asyncio.create_subprocess_exec(
        docker_bin, "image", "inspect", image,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    rc = await proc.wait()
    if rc == 0:
        return
    pull = await asyncio.create_subprocess_exec(
        docker_bin, "pull", image,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await pull.wait()
