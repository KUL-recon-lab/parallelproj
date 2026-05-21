#!/usr/bin/env python3
"""Release helper for parallelproj.

Invoked via ``pixi run release`` (or ``pixi run release-dry-run``).
Reads ``[project].version`` from pyproject.toml, validates that it
looks like a release-grade version string, refuses to proceed if the
corresponding ``v<version>`` tag already exists, and otherwise creates
an annotated tag at HEAD and pushes it (along with the current branch)
to ``origin``.

Use ``--dry-run`` to run all the checks and print what *would* happen,
without creating any tag or pushing anything.

The pre-push hook in ``.githooks/pre-push`` is the final safety net:
it re-reads pyproject.toml at the tagged commit and refuses the push
on mismatch.  This script's job is to make the common path easy and
catch the most common mistakes locally before they reach git.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib


# PEP 440 canonical forms only:
#   X.Y.Z                  - final release
#   X.Y.Z.devN             - development snapshot
#   X.Y.ZaN, .ZbN, .ZrcN   - alpha / beta / release candidate
#
# Strict on purpose: rejecting non-canonical aliases (``alpha1``,
# ``c1``, ``.a1``) and forms we don't intend to publish (``.postN``,
# ``+local``) means the tag list stays consistent and grep-able.  If
# we ever need to broaden this, extend the regex here.
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(\.dev\d+|a\d+|b\d+|rc\d+)?")


def die(*lines: str) -> int:
    """Print error lines to stderr and return exit code 1."""
    for line in lines:
        print(line, file=sys.stderr)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag HEAD with v<[project].version> and push to origin.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks and show what would happen, but don't tag or push.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with open("pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]

    tag = f"v{version}"

    # 1. Version format check.
    if not VERSION_RE.fullmatch(version):
        return die(
            f"ERROR: [project].version {version!r} is not a recognized release format.",
            "       Expected X.Y.Z optionally followed by .devN, aN, bN, or rcN",
            "       (e.g. 2.1.0, 2.1.0.dev0, 2.1.0a1, 2.1.0b1, 2.1.0rc1).",
        )

    # 2. Tag-doesn't-already-exist check (local).  Remote duplicates
    # would be caught by ``git push`` itself, so we don't query the
    # remote here -- avoids a network round-trip in the common case.
    existing = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if existing.returncode == 0:
        return die(
            f"ERROR: tag {tag} already exists locally.",
            f"       Delete it with: git tag -d {tag}",
            "       Or bump [project].version to a new value.",
        )

    # 3. Tag and push -- or, in dry-run mode, just show what would happen.
    if args.dry_run:
        print(f"[DRY RUN] [project].version: {version}")
        print(f"[DRY RUN] Tag to create:     {tag}")
        print("[DRY RUN] Checks passed:     version format OK, tag does not exist")
        print("[DRY RUN] Would execute:")
        print(f"[DRY RUN]   git tag -a {tag} -m {tag}")
        print("[DRY RUN]   git push --follow-tags")
        print("[DRY RUN] No changes made.  Re-run without --dry-run to actually release.")
        return 0

    print(f"Tagging HEAD as {tag} and pushing...")
    subprocess.run(["git", "tag", "-a", tag, "-m", tag], check=True)
    subprocess.run(["git", "push", "--follow-tags"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
