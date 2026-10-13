from pathlib import Path
import re


WORKFLOW = Path(".github/workflows/ci.yml")


def test_integration_job_is_not_conditionally_skipped():
    workflow = WORKFLOW.read_text()
    match = re.search(r"(?ms)^  integration:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", workflow)
    assert match is not None
    assert not re.search(r"(?m)^    if:", match.group("body"))


def test_ci_uses_read_only_token_and_runs_for_merge_queue():
    workflow = WORKFLOW.read_text()
    assert re.search(r"(?m)^  merge_group:\s*$", workflow)
    assert re.search(r"(?ms)^permissions:\n  contents: read\s*$", workflow)


def test_integration_job_uses_public_dependencies_without_secrets():
    workflow = WORKFLOW.read_text()
    match = re.search(r"(?ms)^  integration:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", workflow)
    assert match is not None
    body = match.group("body")
    assert "secrets." not in body
    assert "vars." not in body
    assert "create-github-app-token" not in body
    assert ".[integration]" in body
