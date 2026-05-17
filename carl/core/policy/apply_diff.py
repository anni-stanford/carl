"""Apply a :class:`PolicyDiff` to a :class:`Policy`.

This is the missing link between "CARL proposed a candidate" and "CARL
actually changed your CLAUDE.md". Each operation in
``PolicyDiff.operation`` is implemented here as a pure function over the
policy's artifact list.

All operations are **non-destructive** (they return a new ``Policy``) and
**locality-checked** (the locality budget enforced by
``PolicyDiff.is_within_locality_budget`` should be checked by the caller
before invoking ``apply_diff``).
"""

from __future__ import annotations

from carl.core.policy.artifacts import Artifact, ArtifactType, Policy, PolicyDiff


def apply_diff(policy: Policy, diff: PolicyDiff) -> Policy:
    """Return a new :class:`Policy` with ``diff`` applied.

    The returned policy is suitable for ``adapter.write_policy()``: it has
    a deterministic new ``version`` derived from the parent and the diff,
    a ``parent_version`` pointer for git-tag-based history, and the
    artifact list is mutated according to the diff's operation.
    """
    new_artifacts = [
        Artifact(name=a.name, type=a.type, content=a.content, metadata=dict(a.metadata))
        for a in policy.artifacts
    ]

    target_idx = _find_artifact_index(new_artifacts, diff.artifact_name, diff.artifact_type)
    op = diff.operation

    if op in ("create_skill", "edit_skill"):
        new_content = diff.new_content or ""
        if target_idx is None:
            new_artifacts.append(
                Artifact(name=diff.artifact_name, type=diff.artifact_type, content=new_content)
            )
        else:
            existing = new_artifacts[target_idx]
            new_artifacts[target_idx] = Artifact(
                name=existing.name,
                type=existing.type,
                content=new_content,
                metadata=existing.metadata,
            )

    elif op == "add_line":
        if target_idx is None:
            raise ValueError(
                f"add_line requires existing artifact: {diff.artifact_name} ({diff.artifact_type.value})"
            )
        existing = new_artifacts[target_idx]
        new_artifacts[target_idx] = Artifact(
            name=existing.name,
            type=existing.type,
            content=_apply_add_line(existing.content, diff),
            metadata=existing.metadata,
        )

    elif op == "edit_line":
        if target_idx is None:
            raise ValueError(
                f"edit_line requires existing artifact: {diff.artifact_name}"
            )
        existing = new_artifacts[target_idx]
        new_artifacts[target_idx] = Artifact(
            name=existing.name,
            type=existing.type,
            content=_apply_edit_line(existing.content, diff),
            metadata=existing.metadata,
        )

    elif op == "remove_line":
        if target_idx is None:
            raise ValueError(
                f"remove_line requires existing artifact: {diff.artifact_name}"
            )
        existing = new_artifacts[target_idx]
        new_artifacts[target_idx] = Artifact(
            name=existing.name,
            type=existing.type,
            content=_apply_remove_line(existing.content, diff),
            metadata=existing.metadata,
        )

    elif op == "tighten_setting":
        if target_idx is None:
            raise ValueError(
                f"tighten_setting requires existing artifact: {diff.artifact_name}"
            )
        existing = new_artifacts[target_idx]
        new_artifacts[target_idx] = Artifact(
            name=existing.name,
            type=existing.type,
            content=diff.new_content or existing.content,
            metadata=existing.metadata,
        )

    elif op == "modify_hook":
        new_content = diff.new_content or ""
        if target_idx is None:
            new_artifacts.append(
                Artifact(name=diff.artifact_name, type=ArtifactType.HOOK, content=new_content)
            )
        else:
            existing = new_artifacts[target_idx]
            new_artifacts[target_idx] = Artifact(
                name=existing.name,
                type=ArtifactType.HOOK,
                content=new_content,
                metadata=existing.metadata,
            )

    else:
        raise ValueError(f"unknown PolicyDiff.operation: {op!r}")

    new_version = _derive_version(policy.version, diff)
    return Policy(
        artifacts=new_artifacts,
        version=new_version,
        parent_version=policy.version,
    )


def _find_artifact_index(
    artifacts: list[Artifact], name: str, artifact_type: ArtifactType
) -> int | None:
    for i, a in enumerate(artifacts):
        if a.name == name and a.type == artifact_type:
            return i
    return None


def _apply_add_line(content: str, diff: PolicyDiff) -> str:
    lines = content.splitlines(keepends=False)
    new_text = (diff.new_content or "").rstrip("\n")
    insert_at = diff.line_range[0] - 1 if diff.line_range else len(lines)
    insert_at = max(0, min(insert_at, len(lines)))
    return "\n".join(lines[:insert_at] + [new_text] + lines[insert_at:]) + "\n"


def _apply_edit_line(content: str, diff: PolicyDiff) -> str:
    if diff.line_range is None:
        return diff.new_content or content
    lines = content.splitlines(keepends=False)
    start, end = diff.line_range
    start_idx = max(0, start - 1)
    end_idx = min(len(lines), end)
    new_lines = (diff.new_content or "").rstrip("\n").split("\n") if diff.new_content else []
    return "\n".join(lines[:start_idx] + new_lines + lines[end_idx:]) + "\n"


def _apply_remove_line(content: str, diff: PolicyDiff) -> str:
    if diff.line_range is None:
        return content
    lines = content.splitlines(keepends=False)
    start, end = diff.line_range
    start_idx = max(0, start - 1)
    end_idx = min(len(lines), end)
    return "\n".join(lines[:start_idx] + lines[end_idx:]) + ("\n" if lines[end_idx:] else "")


def _derive_version(parent_version: str, diff: PolicyDiff) -> str:
    """Stable version derived from parent + diff identity, suitable for git tagging."""
    import hashlib

    h = hashlib.sha256(
        f"{parent_version}|{diff.artifact_type.value}|{diff.artifact_name}|{diff.operation}|{diff.new_content or ''}".encode()
    ).hexdigest()[:8]
    return f"{parent_version}+{diff.operation}-{h}"
