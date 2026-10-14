"""Guard World against direct dependencies on human-facing and lab packages."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path


FORBIDDEN_PACKAGE_ROOTS = {"lab", "frontend"}
DEPENDENCY_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _forbidden_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = (alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = (node.module.partition(".")[0],)
        else:
            continue
        found.update(root for root in roots if root in FORBIDDEN_PACKAGE_ROOTS)
    return found


def _forbidden_dependencies(project: dict[str, object]) -> set[str]:
    declarations: list[str] = list(project.get("dependencies", []))  # type: ignore[arg-type]
    optional = project.get("optional-dependencies", {})
    if isinstance(optional, dict):
        for group in optional.values():
            declarations.extend(group)

    found: set[str] = set()
    for declaration in declarations:
        match = DEPENDENCY_NAME.match(declaration)
        if match:
            normalized = re.sub(r"[-_.]+", "-", match.group(1)).lower()
            if normalized in FORBIDDEN_PACKAGE_ROOTS:
                found.add(normalized)
    return found


def test_world_source_has_no_direct_lab_or_frontend_imports() -> None:
    source_root = Path(__file__).parents[1] / "src" / "world"
    violations = {
        f"{path.relative_to(source_root)}: {name}"
        for path in source_root.rglob("*.py")
        for name in _forbidden_imports(path.read_text(encoding="utf-8"))
    }
    assert not violations, "Forbidden direct imports: " + ", ".join(sorted(violations))


def test_world_package_metadata_has_no_direct_lab_or_frontend_dependencies() -> None:
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    with project_file.open("rb") as stream:
        project = tomllib.load(stream)["project"]

    violations = _forbidden_dependencies(project)
    assert not violations, "Forbidden direct dependencies: " + ", ".join(sorted(violations))


def test_forbidden_direct_import_forms_are_detected() -> None:
    cases = {
        "import lab.protocols": {"lab"},
        "from lab import protocol": {"lab"},
        "import frontend.viewer": {"frontend"},
        "from frontend.components import Viewer": {"frontend"},
    }
    for source, expected in cases.items():
        assert _forbidden_imports(source) == expected


def test_forbidden_direct_dependency_forms_are_detected() -> None:
    project = {
        "dependencies": ["lab>=1.0", "frontend[viewer]>=2.0"],
        "optional-dependencies": {
            "integration": [
                "lab @ git+https://example.invalid/lab.git",
                "frontend @ git+https://example.invalid/frontend.git",
            ]
        },
    }
    assert _forbidden_dependencies(project) == FORBIDDEN_PACKAGE_ROOTS
