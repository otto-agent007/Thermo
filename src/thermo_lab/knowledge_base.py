"""Structural checks for the knowledge base under ``docs/knowledge``.

Every source card opens with its title and a fenced ``toml`` block that holds
the card's machine-readable fields and its claims. :func:`check_knowledge_base`
validates those blocks against the repository (pinned dependencies, cited
reports, links, the README index) without network access, so it can run in
CI. :func:`check_upstream` is the separate, networked check for newer
upstream versions; it never runs in tests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.request
from collections.abc import Callable, Iterable
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import Field, ValidationError, model_validator

from thermo_lab.records import FrozenModel

KNOWLEDGE_DIR = Path("docs/knowledge")
_CARD_BLOCK = re.compile(r"\A# (?P<title>[^\n]+)\n\n```toml\n(?P<body>.*?)\n```\n", re.DOTALL)
_LINK = re.compile(r"\]\((?P<target>[^)\s]+)\)")
_ARXIV_URL = re.compile(r"\Ahttps://arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5})(?P<version>v\d+)\Z")
_GITHUB_URL = re.compile(r"\Ahttps://github\.com/(?P<repo>[\w.-]+/[\w.-]+)\Z")
_COMMIT = re.compile(r"\A[0-9a-f]{7,40}\Z")


class ClaimStatus(StrEnum):
    """Thermo's relationship to a source or one of its claims (README, "Claim status")."""

    ASSERTED = "asserted"
    CODE_RELEASED = "code_released"
    PINNED_DEPENDENCY = "pinned_dependency"
    REPRODUCED = "reproduced"


class SourceKind(StrEnum):
    PAPER = "paper"
    LIBRARY = "library"
    BLOG_POST = "blog_post"
    TASK_CATALOG = "task_catalog"


class Source(FrozenModel):
    url: str = Field(pattern=r"^https://")
    version: str = Field(min_length=1)
    released: date | None = None
    authors: str | None = None
    published: str | None = Field(default=None, pattern=r"^https://")
    license: str | None = None
    runtime: str | None = None
    pypi: str | None = None
    note: str | None = None


class Code(FrozenModel):
    """Code that accompanies a paper: an external repository or another card."""

    url: str | None = Field(default=None, pattern=r"^https://")
    card: str | None = None
    license: str | None = None
    note: str | None = None


class Dependency(FrozenModel):
    package: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Claim(FrozenModel):
    id: str = Field(pattern=r"^C[0-9]+$")
    status: ClaimStatus
    text: str = Field(min_length=1)
    reports: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reproduced_cites_reports(self) -> Self:
        if self.status is ClaimStatus.REPRODUCED and not self.reports:
            raise ValueError(f"claim {self.id} is reproduced but cites no report")
        return self


class Card(FrozenModel):
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    kind: SourceKind
    status: ClaimStatus
    last_checked: date
    source: Source
    code: Code | None = None
    dependency: Dependency | None = None
    reports: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.kind in (SourceKind.LIBRARY, SourceKind.TASK_CATALOG):
            if not self.source.license or not self.source.runtime:
                raise ValueError(f"a {self.kind.value} card must record source.license and runtime")
        if (self.status is ClaimStatus.PINNED_DEPENDENCY) != (self.dependency is not None):
            raise ValueError("a [dependency] table is required exactly when status is pinned")
        if self.status is ClaimStatus.REPRODUCED and not self.reports:
            raise ValueError("a reproduced card must cite reports")
        ids = [claim.id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate claim ids: {ids}")
        if "arxiv.org" in self.source.url:
            match = _ARXIV_URL.match(self.source.url)
            if match is None or match["version"] != self.source.version:
                raise ValueError(
                    "an arXiv source must link its exact version, e.g. .../abs/2501.00001v2"
                )
        return self


def parse_card(path: Path) -> Card:
    """Parse one card's header block; raise ``ValueError`` on any defect."""

    text = path.read_text(encoding="utf-8")
    match = _CARD_BLOCK.match(text)
    if match is None:
        raise ValueError("card must start with '# Title', a blank line and a ```toml block")
    try:
        data = tomllib.loads(match["body"])
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in card block: {error}") from error
    try:
        card = Card.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'card'}: {item['msg']}"
            for item in error.errors()
        )
        raise ValueError(problems) from error
    if card.title != match["title"]:
        raise ValueError(f"title {card.title!r} differs from heading {match['title']!r}")
    if card.id != path.stem:
        raise ValueError(f"id {card.id!r} differs from file name {path.stem!r}")
    return card


def _pinned_versions(repo_root: Path) -> tuple[set[str], dict[str, set[str]]]:
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(project["project"]["dependencies"])
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    locked: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        locked.setdefault(package["name"], set()).add(package["version"])
    return declared, locked


def _card_issues(card: Card, path: Path, repo_root: Path, today: date) -> Iterable[str]:
    if card.last_checked > today:
        yield f"last_checked {card.last_checked} is in the future"
    cited = list(card.reports) + [report for claim in card.claims for report in claim.reports]
    for report in cited:
        if not (path.parent / report).resolve().exists():
            yield f"cited report does not exist: {report}"
    if card.code is not None and card.code.card is not None:
        if not (path.parent / f"{card.code.card}.md").is_file():
            yield f"code.card names a missing card: {card.code.card}"
    if card.dependency is not None:
        declared, locked = _pinned_versions(repo_root)
        pin = f"{card.dependency.package}=={card.dependency.version}"
        if pin not in declared:
            yield f"{pin} is not declared in pyproject.toml"
        if card.dependency.version not in locked.get(card.dependency.package, set()):
            yield f"{pin} is not locked in uv.lock"


def _link_issues(markdown: Path) -> Iterable[str]:
    for match in _LINK.finditer(markdown.read_text(encoding="utf-8")):
        target = match["target"].split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (markdown.parent / target).exists():
            yield f"broken link: {match['target']}"


def _index_issues(readme: Path, cards: dict[str, Card]) -> Iterable[str]:
    rows: dict[str, str] = {}
    for line in readme.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        link = re.search(r"\(sources/(?P<stem>[a-z0-9-]+)\.md\)", cells[0])
        if link is None:
            continue
        if link["stem"] in rows:
            yield f"README indexes {link['stem']} twice"
        rows[link["stem"]] = cells[2]
    for stem, card in cards.items():
        if stem not in rows:
            yield f"README index is missing {stem}"
        elif not rows[stem].startswith(f"`{card.status.value}`"):
            yield f"README status for {stem} must start with `{card.status.value}`"
    for stem in rows.keys() - cards.keys():
        yield f"README indexes {stem}, which has no valid card"


def check_knowledge_base(repo_root: Path, *, today: date | None = None) -> list[str]:
    """Return every structural issue in the knowledge base, as ``path: message``."""

    today = today or date.today()
    root = repo_root / KNOWLEDGE_DIR
    issues: list[str] = []
    cards: dict[str, Card] = {}
    for path in sorted((root / "sources").glob("*.md")):
        relative = path.relative_to(repo_root)
        try:
            card = parse_card(path)
        except ValueError as error:
            issues.append(f"{relative}: {error}")
            continue
        cards[card.id] = card
        issues.extend(
            f"{relative}: {issue}" for issue in _card_issues(card, path, repo_root, today)
        )
    for markdown in sorted(root.rglob("*.md")):
        relative = markdown.relative_to(repo_root)
        issues.extend(f"{relative}: {issue}" for issue in _link_issues(markdown))
    readme = root / "README.md"
    issues.extend(
        f"{readme.relative_to(repo_root)}: {issue}" for issue in _index_issues(readme, cards)
    )
    return issues


def load_cards(repo_root: Path) -> list[Card]:
    return [
        parse_card(path) for path in sorted((repo_root / KNOWLEDGE_DIR / "sources").glob("*.md"))
    ]


def stale_cards(cards: Iterable[Card], *, today: date, max_age_days: int) -> list[str]:
    return [
        f"{card.id}: last checked {card.last_checked} ({(today - card.last_checked).days} days ago)"
        for card in cards
        if (today - card.last_checked).days > max_age_days
    ]


Fetch = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "thermo-lab knowledge check"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def check_upstream(cards: Iterable[Card], fetch: Fetch = _fetch) -> list[str]:
    """Report cards whose recorded version is behind upstream. Needs network access."""

    findings: list[str] = []
    for card in cards:
        try:
            latest = _latest_upstream(card, fetch)
        except Exception as error:  # report and continue; one source must not hide the rest
            findings.append(f"{card.id}: upstream check failed: {error}")
            continue
        if latest is not None and latest != card.source.version:
            findings.append(f"{card.id}: card records {card.source.version}, upstream has {latest}")
    return findings


def _latest_upstream(card: Card, fetch: Fetch) -> str | None:
    source = card.source
    if source.pypi is not None:
        data = json.loads(fetch(f"https://pypi.org/pypi/{source.pypi}/json"))
        return str(data["info"]["version"])
    arxiv = _ARXIV_URL.match(source.url)
    if arxiv is not None:
        feed = fetch(f"https://export.arxiv.org/api/query?id_list={arxiv['id']}").decode()
        found = re.search(rf"<id>https?://arxiv\.org/abs/{re.escape(arxiv['id'])}(v\d+)</id>", feed)
        return found.group(1) if found else None
    github = _GITHUB_URL.match(source.url)
    if github is not None and _COMMIT.match(source.version):
        data = json.loads(
            fetch(f"https://api.github.com/repos/{github['repo']}/commits?per_page=1")
        )
        head = str(data[0]["sha"])
        return source.version if head.startswith(source.version) else head[: len(source.version)]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the knowledge base under docs/knowledge.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--stale-after", type=int, metavar="DAYS", help="also list stale cards")
    parser.add_argument("--upstream", action="store_true", help="compare versions upstream")
    args = parser.parse_args(argv)
    issues = check_knowledge_base(args.repo_root)
    for issue in issues:
        print(issue)
    if issues:
        return 1
    cards = load_cards(args.repo_root)
    if args.stale_after is not None:
        for line in stale_cards(cards, today=date.today(), max_age_days=args.stale_after):
            print(f"stale: {line}")
    if args.upstream:
        for line in check_upstream(cards):
            print(f"upstream: {line}")
    print(f"ok: {len(cards)} cards, {sum(len(card.claims) for card in cards)} claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
