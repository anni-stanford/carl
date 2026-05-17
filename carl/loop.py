"""``carl.loop`` — the main async RL loop.

This is the canonical wiring of the four-technique stack:
RLVR (verifier) + GRPO-style group-relative scoring + DPO over policy diffs +
Thompson-sampling bandits, with a paired-bootstrap promotion gate.

Day-1 status: this module is a **typed contract** for the orchestration. The
component implementations land Days 5–11 per the build sequence in the
project specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from carl.adapters.base import PolicyAdapter
    from carl.settings import CARLConfig


class _Stub(Protocol):
    async def _stub(self) -> None: ...  # noqa: D401


async def carl_loop(
    adapters: list[PolicyAdapter],
    config: CARLConfig,
) -> None:
    """Main CARL optimization loop. Implemented Day 12 of the build sequence."""
    raise NotImplementedError(
        "carl.loop.carl_loop is a contract; component wiring lands Days 5–11"
    )
