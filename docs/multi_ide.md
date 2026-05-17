# CARL — Multi-IDE adapter pattern

Every IDE-specific behavior lives in `carl/adapters/<ide>.py`. The core RL loop never touches `open(path, ...)` and never knows whether it's running Claude Code or Cursor.

## Adding a new adapter (≈ 200 LOC)

1. Subclass `PolicyAdapter` in `carl/adapters/<ide>.py`.
2. Implement `read_policy`, `write_policy`, `run_episode`, `list_artifact_types`, `name`.
3. Register the entry point in `pyproject.toml`:
   ```toml
   [project.entry-points."carl.adapters"]
   <ide> = "carl.adapters.<ide>:<Ide>Adapter"
   ```
4. Add a round-trip unit test in `tests/unit/test_<ide>_adapter_round_trip.py`.
5. Open a PR.

## Cursor — Node.js JSON-RPC bridge

The Python `CursorAdapter` spawns `js/cursor-bridge/dist/index.js` as a subprocess. Communication is newline-delimited JSON-RPC 2.0 on stdin/stdout.

Methods (Day 4 of the build sequence implements these):

| Method | Params | Result |
|---|---|---|
| `ping` | — | `{ "pong": true }` |
| `run_episode` | `{ repoPath, prompt, model?, useCloud?, skillsDir, hooksFile, timeoutMs }` | `{ events[], filesChanged[], exitCode, durationS, rawCiOutput? }` |
| `list_models` | — | `{ models: ["composer-2", "claude-opus-4-7", ...] }` |

Why a bridge instead of HTTP API? `@cursor/sdk` is the official integration path; it manages cloud VMs and local agent processes for us. Re-implementing that on top of raw HTTP would be brittle and slower to track Cursor's changes.

## Cloud vs local sandbox

When `CARL_USE_CLOUD=1`, episodes run on Cursor's cloud VMs (managed by the SDK). When `0`, they run in local Docker containers built from `docker/Dockerfile.episode.{claude,cursor}`. The `Trajectory` returned is identical; only the runtime differs.
