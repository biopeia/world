from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from world.m1_profile import ProfileError, _validate_profile, load_operator_profile

PROFILE = Path(__file__).parents[1] / "profiles/m1/scalar-sensorimotor-mujoco-3.15-v2.json"


def _profile() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_bundled_operator_profile_is_explicit_and_strictly_valid() -> None:
    profile, digest = load_operator_profile(PROFILE)
    assert profile["profile_kind"] == "operator_supplied_scenario"
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(unexpected=True), "fields mismatch"),
        (lambda p: p["environment"].update(gravity_m_s2=[0, 0]), "3-element array"),
        (lambda p: p["environment"]["field"].update(field_radius_m=-1), ">= 0"),
        (lambda p: p["modality"].update(sense_node_id="different.node"), "must match"),
        (lambda p: p["embodiment"]["joints"][0].update(axis=[]), "3-element array"),
        (lambda p: p["embodiment"]["torque_binding"].update(joint_id="other.joint"), "target the declared joint"),
    ],
)
def test_profile_rejects_malformed_or_incoherent_values(mutate, message: str) -> None:
    candidate = copy.deepcopy(_profile())
    mutate(candidate)
    with pytest.raises(ProfileError, match=message):
        _validate_profile(candidate)


def test_loader_rejects_unknown_schema_and_duplicate_keys(tmp_path: Path) -> None:
    raw = json.dumps(_profile()).replace('"schema_version": 1', '"schema_version": 2')
    path = tmp_path / "unsupported.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ProfileError, match="schema/version"):
        load_operator_profile(path)
    path.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(ProfileError, match="duplicate"):
        load_operator_profile(path)


def test_loader_rejects_symlinked_profile(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    link = tmp_path / "profile.json"
    link.symlink_to(target)
    with pytest.raises(ProfileError, match="non-symlink"):
        load_operator_profile(link)
