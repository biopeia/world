"""Composition boundary for Environment-owned checkpoint restoration.

This optional integration adapter keeps Environment state authoritative while
passing World's independently committed checkpoint identity and digest to the
Environment's native restored-owner verifier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from environment import DynamicEnvironmentOwner, OwnerRestorationReference

from world.manifest import CheckpointReference


class DynamicEnvironmentCheckpointContract:
    """Validate and verify one restored dynamic Environment owner.

    The owner restorer attaches the restored Environment instance after it has
    been reconstructed from the referenced checkpoint. Verification uses the
    digest and identity fields from World's committed CheckpointReference; the
    Environment-native verifier then compares the restored instance's canonical
    checkpoint bytes against that independent digest.
    """

    def __init__(self) -> None:
        self._restored_owner: DynamicEnvironmentOwner | None = None

    def attach_restored_owner(self, owner: DynamicEnvironmentOwner) -> None:
        if not isinstance(owner, DynamicEnvironmentOwner):
            raise TypeError("restored owner must be a DynamicEnvironmentOwner")
        self._restored_owner = owner

    def validate_checkpoint(
        self,
        encoded: bytes,
        owner_identity: str,
        schema_version: str,
    ) -> bool:
        return DynamicEnvironmentOwner.validate_checkpoint(
            encoded, owner_identity, schema_version
        )

    def verify_restoration(self, reference: CheckpointReference) -> bool:
        owner = self._restored_owner
        if owner is None:
            return False
        if reference.owner != "environment":
            return False

        # World references currently carry owner identity, schema, locator and
        # digest, but not Environment-specific bindings. Derive those bindings
        # from the committed bytes, never from the restored candidate. World
        # verifies the same digest before calling this boundary; repeat it here
        # to keep this adapter safe against a changed referenced file.
        try:
            encoded = Path(reference.locator).read_bytes()
            if hashlib.sha256(encoded).hexdigest() != reference.sha256:
                return False
            committed_owner = DynamicEnvironmentOwner.from_checkpoint_bytes(encoded)
        except (OSError, TypeError, ValueError):
            return False

        if (committed_owner.owner_id, committed_owner.CHECKPOINT_SCHEMA) != (
            reference.owner_identity,
            reference.schema_version,
        ):
            return False

        owner_reference = OwnerRestorationReference(
            owner_identity=reference.owner_identity,
            schema_version=reference.schema_version,
            checkpoint_sha256=reference.sha256,
            environment_id=committed_owner.environment_id,
            environment_revision=committed_owner.environment_revision,
            world_id=committed_owner.world_id,
        )
        try:
            owner.verify_restoration(owner_reference)
        except (TypeError, ValueError):
            return False
        return True


__all__ = ["DynamicEnvironmentCheckpointContract"]
