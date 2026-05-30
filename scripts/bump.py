#!/usr/bin/env python3
"""Bump the version, update the changelog, commit, and tag a release.

Usage:
    python scripts/bump.py patch          # 0.1.0 -> 0.1.1
    python scripts/bump.py minor          # 0.1.0 -> 0.2.0
    python scripts/bump.py major          # 0.1.0 -> 1.0.0
    python scripts/bump.py 0.5.0          # set an explicit version
    python scripts/bump.py patch --push   # also push main + the tag (triggers the PyPI publish)

What it does, atomically:
  1. Bumps `version` in pyproject.toml AND `__version__` in the package __init__
     (the two must never drift apart).
  2. Promotes the CHANGELOG "## Unreleased" section to "## X.Y.Z (today)" and
     opens a fresh empty "## Unreleased" above it.
  3. Commits "release: vX.Y.Z" and creates an annotated tag vX.Y.Z.

Pushing the tag is what triggers .github/workflows/release.yml, which tests,
builds, and publishes to PyPI via trusted publishing (no token).
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "langchain_failover" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def run(*args: str) -> str:
    return subprocess.run(
        args, cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout


def die(msg: str) -> "None":
    sys.exit(f"bump: {msg}")


def current_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.M)
    if not m:
        die("could not find `version = \"...\"` in pyproject.toml")
    return m.group(1)


def next_version(cur: str, part: str) -> str:
    m = SEMVER.match(cur)
    if not m:
        die(f"current version {cur!r} is not X.Y.Z")
    major, minor, patch = (int(x) for x in m.groups())
    return {
        "major": f"{major + 1}.0.0",
        "minor": f"{major}.{minor + 1}.0",
        "patch": f"{major}.{minor}.{patch + 1}",
    }[part]


def ensure_releasable() -> None:
    dirty = run("git", "status", "--porcelain").strip()
    if dirty:
        die("working tree is not clean; commit or stash first:\n" + dirty)
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD").strip()
    if branch != "main":
        die(f"on branch {branch!r}; release from main")


def replace_once(path: pathlib.Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if n != 1:
        die(f"expected exactly one match for {pattern!r} in {path.name} (found {n})")
    path.write_text(new)


def roll_changelog(new: str) -> None:
    text = CHANGELOG.read_text()
    today = datetime.date.today().isoformat()
    if re.search(r"^##\s+Unreleased\s*$", text, re.M | re.I):
        # Promote Unreleased -> the new version, and open a fresh Unreleased above.
        text = re.sub(
            r"^##\s+Unreleased\s*$",
            f"## Unreleased\n\n## {new} ({today})",
            text,
            count=1,
            flags=re.M | re.I,
        )
    else:
        # No Unreleased section: insert a stub right under the top heading.
        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.lstrip().lower().startswith("# changelog"):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                lines.insert(j, f"## {new} ({today})\n\n- TODO: describe changes.\n\n")
                break
        else:
            lines = [f"# Changelog\n\n## {new} ({today})\n\n- TODO: describe changes.\n\n", *lines]
        text = "".join(lines)
    CHANGELOG.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bump version, update changelog, commit, tag.")
    ap.add_argument("version", help="patch | minor | major | explicit X.Y.Z")
    ap.add_argument("--push", action="store_true",
                    help="also push main and the tag (triggers the PyPI publish workflow)")
    args = ap.parse_args()

    ensure_releasable()
    cur = current_version()
    if args.version in ("patch", "minor", "major"):
        new = next_version(cur, args.version)
    elif SEMVER.match(args.version):
        new = args.version
    else:
        die("version must be patch|minor|major or X.Y.Z")
    if new == cur:
        die(f"new version equals current ({cur})")

    print(f"{cur} -> {new}")
    replace_once(PYPROJECT, r'^version\s*=\s*"[^"]+"', f'version = "{new}"')
    replace_once(INIT, r'^__version__\s*=\s*"[^"]+"', f'__version__ = "{new}"')
    roll_changelog(new)

    run("git", "add", "pyproject.toml", "src/langchain_failover/__init__.py", "CHANGELOG.md")
    run("git", "commit", "-m", f"release: v{new}")
    run("git", "tag", "-a", f"v{new}", "-m", f"langchain-failover {new}")
    print(f"committed + tagged v{new}")

    if args.push:
        run("git", "push", "origin", "main")
        run("git", "push", "origin", f"v{new}")
        print(f"pushed main + v{new} — the Release workflow will publish to PyPI")
    else:
        print(f"next:  git push origin main && git push origin v{new}")


if __name__ == "__main__":
    main()
