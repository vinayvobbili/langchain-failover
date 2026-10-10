"""scripts/bump.py: the changelog roll and version arithmetic."""
from __future__ import annotations

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "bump.py"
_spec = importlib.util.spec_from_file_location("bump", _PATH)
assert _spec and _spec.loader
bump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bump)


def test_next_version():
    assert bump.next_version("0.2.0", "patch") == "0.2.1"
    assert bump.next_version("0.2.9", "minor") == "0.3.0"
    assert bump.next_version("0.2.9", "major") == "1.0.0"


def test_roll_keeps_the_blank_line_under_the_new_heading(tmp_path, monkeypatch):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## Unreleased\n\n- Docs.\n\n## 0.2.0 (2026-06-28)\n")
    monkeypatch.setattr(bump, "CHANGELOG", changelog)
    bump.roll_changelog("0.2.1")
    text = changelog.read_text()
    assert text.startswith("# Changelog\n\n## Unreleased\n\n## 0.2.1 (")
    assert ")\n\n- Docs.\n\n## 0.2.0 (2026-06-28)\n" in text
