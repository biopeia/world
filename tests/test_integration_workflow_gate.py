from pathlib import Path
import re
import subprocess
import textwrap


WORKFLOW = Path(".github/workflows/ci.yml")


def _credential_preflight() -> str:
    workflow = WORKFLOW.read_text()
    start = workflow.index("      - name: Require integration read credentials")
    token_step = workflow.index("      - uses: actions/create-github-app-token@v2")
    validation = workflow[start:token_step]
    script = re.search(r"(?m)^        run: \|\n((?:^          .*\n)+)", validation)
    assert script is not None
    return textwrap.dedent(script.group(1))


def test_integration_job_is_not_conditionally_skipped():
    workflow = WORKFLOW.read_text()
    match = re.search(r"(?ms)^  integration:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", workflow)
    assert match is not None
    assert not re.search(r"(?m)^    if:", match.group("body"))


def test_missing_app_id_fails_with_value_free_diagnostic():
    result = subprocess.run(
        ["bash", "-e", "-c", _credential_preflight()],
        env={"PATH": "/usr/bin:/bin", "CI_READ_APP_PRIVATE_KEY": "must-not-leak"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "CI_READ_APP_ID is missing" in result.stdout
    assert "must-not-leak" not in result.stdout + result.stderr


def test_missing_private_key_fails_with_value_free_diagnostic():
    result = subprocess.run(
        ["bash", "-e", "-c", _credential_preflight()],
        env={"PATH": "/usr/bin:/bin", "CI_READ_APP_ID": "12345"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "CI_READ_APP_PRIVATE_KEY is missing" in result.stdout
    assert "12345" not in result.stdout + result.stderr
