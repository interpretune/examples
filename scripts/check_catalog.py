"""Validate catalog.yaml: every entry names a pinned, tested component, and nothing else is accepted.

Each field is checked and an unknown one is refused by name, so a typo cannot silently drop a field the scheduled run
relies on. A ``first_party`` entry must have its source under ``components/``.

Usage: ``python scripts/check_catalog.py catalog.yaml``; exits non-zero naming every problem found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REQUIRED = {
    "repo": "the Hub repository, owner/name",
    "revision": "a full 40-character commit sha, never a branch",
    "kinds": "a non-empty list of component kinds",
    "interpretune": "the tested interpretune version range",
    "owner": "who fixes the entry when it breaks",
    "smoke": "what the scheduled run executes: a module configuration key, or a snapshot directory of CPU tests ending in /",
    "first_party": "true when the source lives under components/",
}
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA = re.compile(r"^[0-9a-f]{40}$")
# Kinds that only define things other components compose (compose_ref), so there is no configuration of their own to
# run. An entry made only of these may set ``smoke: null``; any other entry must name its smoke configuration.
DEFINITIONS_ONLY = {"promptconfigs"}


def check_entry(i: int, entry: object, root: Path) -> list[str]:
    """Problems with catalog entry ``i``, each naming the field and what is accepted."""
    where = f"components[{i}]"
    if not isinstance(entry, dict):
        return [f"{where}: an entry must be a mapping, got {type(entry).__name__}"]
    problems = [f"{where}: unknown field {k!r} (accepted: {', '.join(REQUIRED)})" for k in entry if k not in REQUIRED]
    problems += [f"{where}: missing {k!r}: {why}" for k, why in REQUIRED.items() if k not in entry]
    repo, rev, kinds = entry.get("repo"), entry.get("revision"), entry.get("kinds")
    if "repo" in entry and not (isinstance(repo, str) and REPO.match(repo)):
        problems.append(f"{where}: repo {repo!r} is not owner/name")
    if "revision" in entry and not (isinstance(rev, str) and SHA.match(rev)):
        problems.append(f"{where}: revision {rev!r} is not a full 40-character sha")
    if "kinds" in entry and not (isinstance(kinds, list) and kinds and all(isinstance(k, str) and k for k in kinds)):
        problems.append(f"{where}: kinds must be a non-empty list of names")
    definitions_only = isinstance(kinds, list) and bool(kinds) and set(kinds) <= DEFINITIONS_ONLY
    for k in ("interpretune", "owner", "smoke"):
        if k == "smoke" and entry.get(k) is None and definitions_only:
            continue  # nothing of its own to run: other components' configurations compose it
        if k in entry and not (isinstance(entry[k], str) and entry[k].strip()):
            hint = (
                f" (null is accepted only when every kind is one of {sorted(DEFINITIONS_ONLY)})" if k == "smoke" else ""
            )
            problems.append(f"{where}: {k} must be a non-empty string{hint}")
    fp = entry.get("first_party")
    if "first_party" in entry and not isinstance(fp, bool):
        problems.append(f"{where}: first_party must be true or false")
    if fp is True and isinstance(repo, str) and "/" in repo:
        name = repo.split("/", 1)[1]
        if not (root / "components" / name).is_dir():
            problems.append(f"{where}: first_party, but components/{name}/ does not exist")
        elif isinstance(entry.get("smoke"), str):
            problems += check_first_party_smoke(where, root / "components" / name, entry["smoke"])
    return problems


def check_first_party_smoke(where: str, source: Path, smoke: str) -> list[str]:
    """A first-party entry's smoke must name something its source has: a test directory, or a manifest configuration.

    The source is what was published at the entry's revision (``components/README.md``), so a smoke the source cannot
    satisfy would fail only in the scheduled run, a day later.
    """
    if smoke.endswith("/"):
        if not (source / smoke).is_dir():
            return [f"{where}: smoke {smoke!r} names a test directory, but {source.name}/{smoke} does not exist"]
        return []
    manifest = yaml.safe_load((source / "it_component.yaml").read_text()) or {}
    configs = (manifest.get("module") or {}).get("configs") or {}
    if smoke not in configs:
        known = ", ".join(configs) or "none"
        return [
            f"{where}: smoke {smoke!r} is not a module configuration of {source.name}/it_component.yaml (it has: "
            f"{known}); a component without one names a test directory ending in '/'"
        ]
    return []


def check(path: Path) -> list[str]:
    """Every problem in the catalog at ``path``."""
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or set(doc) != {"components"}:
        return [f"{path}: the document must be a mapping with exactly one key, 'components'"]
    entries = doc["components"]
    if not isinstance(entries, list):
        return [f"{path}: 'components' must be a list"]
    problems = [p for i, e in enumerate(entries) for p in check_entry(i, e, path.resolve().parent)]
    repos = [e.get("repo") for e in entries if isinstance(e, dict)]
    problems += [
        f"{path}: repo {r!r} is listed more than once" for r in sorted({r for r in repos if repos.count(r) > 1})
    ]
    return problems


def main(argv: list[str]) -> int:
    """Validate each catalog named on the command line."""
    if not argv:
        print("usage: check_catalog.py CATALOG [CATALOG ...]", file=sys.stderr)
        return 2
    problems = [p for a in argv for p in check(Path(a))]
    for p in problems:
        print(p, file=sys.stderr)
    if not problems:
        print(f"catalog OK: {', '.join(argv)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
