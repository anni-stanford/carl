"""Adapter discovery via ``carl.adapters`` entry points (declared in pyproject.toml)."""

from __future__ import annotations

from importlib.metadata import entry_points

from carl.adapters.base import PolicyAdapter

_GROUP = "carl.adapters"


def load_adapters() -> dict[str, PolicyAdapter]:
    """Instantiate every adapter registered under the ``carl.adapters`` entry-point group."""
    out: dict[str, PolicyAdapter] = {}
    for ep in entry_points(group=_GROUP):
        cls = ep.load()
        adapter: PolicyAdapter = cls()
        out[adapter.name()] = adapter
    return out
