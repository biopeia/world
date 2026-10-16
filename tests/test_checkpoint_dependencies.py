"""Owner checkpoint dependencies (organism#183, world#90).

World persists owner-declared immutable dependency bytes before the payload,
records ``dependency_sha256``/``dependency_parent`` (content references, no
locator) and refuses commits or restores that break the invariants. A fake
opaque owner stands in for Organism: only its own contract reads its bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import world.durable_runtime as durable_runtime_module
from world import (
    CheckpointReference,
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    ManifestError,
    PresenceId,
    PresenceOwnerLink,
    SingleWriterPreconditionError,
    WorldCausalManifestStore,
    WorldGenesisSpec,
    WorldId,
)
from world.manifest import _manifest_to_data, reference_to_data
from world.owner_checkpoint_store import (
    FileOwnerCheckpointStore,
    OwnerCheckpointDependency,
    OwnerCheckpointPayload,
)

OWNER = ("organism", "organism-identity")
WORLD = WorldId("world-dependencies")


# --------------------------------------------------------------- fake owner
def owner_bytes(state: int, head: str | None) -> bytes:
    """The fake owner's private format; World never parses it."""
    return b"fake-owner|" + json.dumps({"state": state, "head": head}, sort_keys=True).encode()


def segment(parent: str | None, n: int) -> bytes:
    return json.dumps({"parent": parent, "n": n}, sort_keys=True).encode()


class DeclaringContract:
    """Owner contract that states which dependency its bytes require."""

    def validate_checkpoint(self, encoded, owner_identity, schema_version):
        return encoded.startswith(b"fake-owner|")

    def verify_restoration(self, reference):
        return self.validate_checkpoint(Path(reference.locator).read_bytes(), "", "")

    def declared_dependency(self, encoded, owner_identity, schema_version):
        return json.loads(encoded.split(b"|", 1)[1])["head"]


class PlainContract(DeclaringContract):
    declared_dependency = None   # an owner that has no dependency support


def verifier(references):
    return all(Path(r.locator).is_file() for r in references)


class Harness:
    def __init__(self, root: Path, contract=None):
        self.root = root
        self.contract = contract or DeclaringContract()
        self.store = FileOwnerCheckpointStore(root / "owners")
        self.manifest_store = WorldCausalManifestStore(root / "manifest.json")
        self.journal_store = DurableJournalSnapshotStore(root / "journal", WORLD)
        self.head: str | None = None
        self.n = 0
        [genesis] = self.store.write_genesis((OwnerCheckpointPayload(*OWNER, "v5", owner_bytes(0, None)),))
        self.runtime = DurableWorldRuntime.initialize(
            WorldGenesisSpec(
                world_id=WORLD, initial_presence_ids=("presence-a",), registered_owner_ids=(OWNER,),
                initial_owner_references=(genesis,), owner_checkpoint_contracts={OWNER: self.contract},
                presence_owner_links=(PresenceOwnerLink(PresenceId("presence-a"), *OWNER),),
            ),
            "presence-a", self.manifest_store, self.journal_store, owner_checkpoint_verifier=verifier,
        )

    def payload(self, *, new=True, head=None, parent=None, declare_head="same", blob=None):
        """One owner checkpoint; defaults produce a valid chained dependency."""
        if new:
            blob = blob if blob is not None else segment(self.head, self.n + 1)
            head = hashlib.sha256(blob).hexdigest() if head is None else head
            parent = self.head if parent is None else parent
            dep = OwnerCheckpointDependency.__new__(OwnerCheckpointDependency)   # allow forged cases
            for k, v in (("head", head), ("parent", parent), ("new_blob", blob)):
                object.__setattr__(dep, k, v)
        else:
            dep = OwnerCheckpointDependency(self.head, self.head) if self.head else None
            head = self.head
        declared = head if declare_head == "same" else declare_head
        return OwnerCheckpointPayload(*OWNER, "v5", owner_bytes(self.n + 1, declared), dep), head

    def tick(self, payload_and_head, transition=None):
        payload, head = payload_and_head
        self.n += 1
        refs = []

        def write(tick):
            refs[:] = self.store.write_genesis((payload,))
            return tuple(refs)

        self.runtime.tick(transition or f"t{self.n}", write)
        self.head = head
        return refs[0]

    def restore(self):
        return DurableWorldRuntime.restore(
            WORLD, "presence-a", self.manifest_store, self.journal_store,
            owner_checkpoint_verifier=verifier, owner_checkpoint_contracts={OWNER: self.contract},
            owner_checkpoint_restorer=lambda refs: None,
        )


def committed_owner_ref(h: Harness) -> CheckpointReference:
    return next(r for r in h.runtime.manifest.checkpoint_references if r.owner == OWNER[0])


# ------------------------------------------------------------ compatibility
def test_references_without_dependency_serialize_exactly_as_before():
    ref = CheckpointReference("o", "i", "c", "v4", "/x/aa/b.blob", "a" * 64)
    legacy = {"owner": "o", "owner_identity": "i", "checkpoint_id": "c", "schema_version": "v4",
              "locator": "/x/aa/b.blob", "sha256": "a" * 64}
    assert reference_to_data(ref) == legacy
    assert durable_runtime_module._reference_descriptor(ref) == {
        "owner_id": "o", "owner_identity": "i", "checkpoint_id": "c", "schema_version": "v4",
        "locator": "/x/aa/b.blob", "sha256": "a" * 64}


def test_dependency_fields_validate():
    with pytest.raises(ValueError):
        CheckpointReference("o", "i", "c", "v", "/l", "a" * 64, dependency_sha256="xyz")
    with pytest.raises(ValueError, match="requires a dependency"):
        CheckpointReference("o", "i", "c", "v", "/l", "a" * 64, dependency_parent="b" * 64)
    with pytest.raises(ValueError, match="do not match"):
        OwnerCheckpointDependency("a" * 64, None, b"bytes")
    with pytest.raises(ValueError, match="keep its parent"):
        OwnerCheckpointDependency("a" * 64, None)


# ------------------------------------------------------------------ commit
def test_chained_dependencies_commit_and_restore(tmp_path):
    h = Harness(tmp_path)
    first = h.tick(h.payload())
    assert first.dependency_parent is None and first.dependency_sha256 == h.head
    second = h.tick(h.payload())
    assert second.dependency_parent == first.dependency_sha256
    unchanged = h.tick(h.payload(new=False))           # step that makes nothing new
    assert unchanged.dependency_sha256 == unchanged.dependency_parent == second.dependency_sha256
    restored = h.restore()
    ref = committed_owner_ref(type("R", (), {"runtime": restored})())
    assert ref.dependency_sha256 == second.dependency_sha256
    assert h.store.resolve(ref.dependency_sha256) == segment(first.dependency_sha256, 2)
    # journal and manifest carry the content references, no dependency locator
    data = _manifest_to_data(restored.manifest)
    owner = next(r for r in data["checkpoint_references"] if r["owner"] == OWNER[0])
    assert set(owner) == {"owner", "owner_identity", "checkpoint_id", "schema_version", "locator", "sha256",
                          "dependency_sha256", "dependency_parent"}


def test_dependency_bytes_are_durable_before_the_payload(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    order = []
    original = FileOwnerCheckpointStore._publish_blob

    def record(self, target, encoded, digest):
        order.append(digest)
        return original(self, target, encoded, digest)

    monkeypatch.setattr(FileOwnerCheckpointStore, "_publish_blob", record)
    payload, head = h.payload()
    h.tick((payload, head))
    assert order == [head, hashlib.sha256(payload.encoded).hexdigest()]


@pytest.mark.parametrize("case", ["omitted", "inconsistent"])
def test_v5_cannot_commit_with_an_omitted_or_inconsistent_dependency(tmp_path, case):
    h = Harness(tmp_path)
    h.tick(h.payload())
    if case == "omitted":
        payload = OwnerCheckpointPayload(*OWNER, "v5", owner_bytes(9, "c" * 64))   # bytes need c..., none attached
        head = None
    else:
        payload, head = h.payload(declare_head="d" * 64)                           # attaches X, bytes need d...
    before = h.runtime.manifest
    with pytest.raises(ManifestError, match="omitted or inconsistent"):
        h.tick((payload, head))
    stored = h.manifest_store.read()
    assert stored.causal_tick == before.causal_tick and stored.commit_id == before.commit_id   # cut not advanced
    assert stored.status.name == "RECOVERY_REQUIRED"                                          # explicit recovery


def test_owner_without_dependency_support_cannot_attach_one(tmp_path):
    h = Harness(tmp_path, contract=PlainContract())
    with pytest.raises(ManifestError, match="does not declare"):
        h.tick(h.payload())


def test_parent_must_extend_the_committed_base_cut(tmp_path):
    h = Harness(tmp_path)
    h.tick(h.payload())
    stale_parent = "e" * 64
    with pytest.raises(ManifestError, match="does not extend"):
        h.tick(h.payload(parent=stale_parent))


def test_declared_head_must_be_durable(tmp_path):
    h = Harness(tmp_path)
    ghost = "f" * 64
    dep = OwnerCheckpointDependency.__new__(OwnerCheckpointDependency)
    for k, v in (("head", ghost), ("parent", None), ("new_blob", None)):
        object.__setattr__(dep, k, v)
    payload = OwnerCheckpointPayload(*OWNER, "v5", owner_bytes(1, ghost), dep)
    with pytest.raises(LookupError):
        h.tick((payload, ghost))


def test_stale_writer_cannot_commit_even_with_valid_dependency(tmp_path):
    h = Harness(tmp_path)
    h.tick(h.payload())
    other = h.restore()                  # a second writer on the same cut
    h.tick(h.payload())                  # first writer advances the cut
    h.runtime, saved = other, h.runtime
    with pytest.raises(SingleWriterPreconditionError):
        h.tick(h.payload())
    h.runtime = saved


# ------------------------------------------------------------------ restore
@pytest.mark.parametrize("damage", ["missing", "corrupted"])
def test_restore_refuses_missing_or_corrupted_dependency(tmp_path, damage):
    h = Harness(tmp_path)
    ref = h.tick(h.payload())
    blob = Path(ref.locator).parent.parent / ref.dependency_sha256[:2] / f"{ref.dependency_sha256}.blob"
    if damage == "missing":
        blob.unlink()
    else:
        blob.chmod(0o600)
        blob.write_bytes(blob.read_bytes() + b"x")
    with pytest.raises(ManifestError, match="dependency closure"):
        h.restore()


def test_dependency_digest_is_location_independent(tmp_path):
    refs = []
    for root in ("a", "b/c"):
        h = Harness(tmp_path / root)
        h.tick(h.payload())
        refs.append(h.tick(h.payload()))
    a, b = refs
    assert (a.dependency_sha256, a.dependency_parent, a.sha256) == (b.dependency_sha256, b.dependency_parent, b.sha256)
    assert a.locator != b.locator      # the existing locator field is unchanged (world#90)


# ------------------------------------------------------------------ genesis
def _genesis(tmp_path, payload):
    store = FileOwnerCheckpointStore(tmp_path / "owners")
    [ref] = store.write_genesis((payload,))
    return DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=WORLD, initial_presence_ids=("presence-a",), registered_owner_ids=(OWNER,),
            initial_owner_references=(ref,), owner_checkpoint_contracts={OWNER: DeclaringContract()},
            presence_owner_links=(PresenceOwnerLink(PresenceId("presence-a"), *OWNER),),
        ),
        "presence-a", WorldCausalManifestStore(tmp_path / "manifest.json"),
        DurableJournalSnapshotStore(tmp_path / "journal", WORLD), owner_checkpoint_verifier=verifier,
    )


def test_genesis_dependency_is_verified_like_any_commit(tmp_path):
    blob = segment(None, 1)
    head = hashlib.sha256(blob).hexdigest()
    with pytest.raises(ManifestError, match="omitted or inconsistent"):      # bytes need head, none attached
        _genesis(tmp_path / "omitted", OwnerCheckpointPayload(*OWNER, "v5", owner_bytes(0, head)))
    other = segment(None, 2)
    other_head = hashlib.sha256(other).hexdigest()
    with pytest.raises(ManifestError, match="does not extend"):              # genesis cannot have a parent
        _genesis(tmp_path / "parented", _with_parent(head, blob, other_head))
    runtime = _genesis(tmp_path / "ok", OwnerCheckpointPayload(
        *OWNER, "v5", owner_bytes(0, head), OwnerCheckpointDependency(head, None, blob)))
    ref = next(r for r in runtime.manifest.checkpoint_references if r.owner == OWNER[0])
    assert ref.dependency_sha256 == head and ref.dependency_parent is None


def _with_parent(head, blob, parent):
    dep = OwnerCheckpointDependency.__new__(OwnerCheckpointDependency)
    for k, v in (("head", head), ("parent", parent), ("new_blob", blob)):
        object.__setattr__(dep, k, v)
    return OwnerCheckpointPayload(*OWNER, "v5", owner_bytes(0, head), dep)


# ------------------------------------------------ review follow-ups (BAW-004)
def _bypass_reference(h: Harness, head: str):
    """A writer that skips the store and points at a dependency never written."""
    payload_bytes = owner_bytes(h.n + 1, head)
    path = h.root / "manual" / "payload.blob"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload_bytes)
    store_layout = h.store.root / hashlib.sha256(payload_bytes).hexdigest()[:2]
    store_layout.mkdir(parents=True, exist_ok=True)
    target = store_layout / f"{hashlib.sha256(payload_bytes).hexdigest()}.blob"
    target.write_bytes(payload_bytes)
    return CheckpointReference(*OWNER, "manual", "v5", str(target), hashlib.sha256(payload_bytes).hexdigest(),
                               dependency_sha256=head, dependency_parent=h.head)


def test_world_commit_check_refuses_a_dependency_that_was_never_written(tmp_path):
    h = Harness(tmp_path)
    h.tick(h.payload())
    before = h.runtime.manifest
    ghost = "9" * 64
    with pytest.raises(ManifestError, match="dependency closure"):   # World's own check, not the store's
        h.runtime.tick("bypass", lambda _t: (_bypass_reference(h, ghost),))
    stored = h.manifest_store.read()
    assert (stored.causal_tick, stored.commit_id) == (before.causal_tick, before.commit_id)
    assert stored.status.name == "RECOVERY_REQUIRED"


def test_recover_previous_cut_with_dependencies(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    committed = h.tick(h.payload())
    with pytest.raises(ManifestError):
        h.tick(h.payload(parent="e" * 64))                      # failed tick -> RECOVERY_REQUIRED

    def recover():
        return DurableWorldRuntime.recover_previous_cut(
            WORLD, "presence-a", h.manifest_store, h.journal_store, owner_checkpoint_verifier=verifier,
            owner_checkpoint_contracts={OWNER: h.contract}, owner_checkpoint_restorer=lambda refs: None)

    blob = h.store.root / committed.dependency_sha256[:2] / f"{committed.dependency_sha256}.blob"
    saved = blob.read_bytes()
    blob.unlink()
    with pytest.raises(ManifestError):                          # base dependency missing: refused
        recover()
    blob.write_bytes(saved)
    h.runtime = recover()
    assert committed_owner_ref(h).dependency_sha256 == committed.dependency_sha256
    h.tick(h.payload())                                         # continues from the recovered cut


@pytest.mark.parametrize("contract", [PlainContract(), type("Liar", (DeclaringContract,), {
    "declared_dependency": lambda self, e, i, s: "a" * 64})()], ids=["no-hook", "contradicting"])
def test_restore_rechecks_the_owner_declaration(tmp_path, contract):
    h = Harness(tmp_path)
    h.tick(h.payload())
    h.contract = contract
    with pytest.raises(ManifestError, match="dependency closure"):
        h.restore()


def test_null_dependency_keys_are_not_a_second_encoding(tmp_path):
    from world.manifest import _manifest_from_data
    h = Harness(tmp_path)
    data = _manifest_to_data(h.runtime.manifest)
    plain = next(r for r in data["checkpoint_references"] if "dependency_sha256" not in r)
    plain.update(dependency_sha256=None, dependency_parent=None)
    with pytest.raises(ValueError, match="empty dependency"):
        _manifest_from_data(data)
