"""Content-addressed resolution of owner checkpoint dependencies (organism#183).

Dependencies are immutable bytes declared by a semantic owner and stored in
the same content-addressed layout as owner checkpoints. World resolves them by
digest only and never interprets their content.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_HEX64 = frozenset("0123456789abcdef")


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def resolve_dependency(root: str | os.PathLike[str], digest: str) -> bytes:
    """Read the content-addressed blob ``digest`` under ``root`` and verify it."""
    if not is_sha256(digest):
        raise LookupError("dependency digest must be a sha256 hex digest")
    path = Path(root) / digest[:2] / f"{digest}.blob"
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LookupError(f"dependency {digest} is unavailable") from exc
    if hashlib.sha256(data).hexdigest() != digest:
        raise LookupError(f"dependency {digest} does not match its digest")
    return data


def reference_dependency_root(reference) -> Path:
    """Store root of a content-addressed owner checkpoint reference."""
    return Path(reference.locator).parent.parent


__all__ = ["is_sha256", "reference_dependency_root", "resolve_dependency"]
