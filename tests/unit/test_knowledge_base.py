import json
from datetime import date
from pathlib import Path

import pytest

from thermo_lab.knowledge_base import (
    check_knowledge_base,
    check_upstream,
    load_cards,
    parse_card,
    stale_cards,
)

ROOT = Path(__file__).parents[2]
TODAY = date(2026, 10, 1)

CARD = """# Example library

```toml
id = "example-lib"
title = "Example library"
kind = "library"
status = "pinned_dependency"
last_checked = 2026-09-29

[source]
url = "https://github.com/example/lib"
version = "1.2.3"
license = "Apache-2.0"
runtime = "JAX"
pypi = "example-lib"

[dependency]
package = "example-lib"
version = "1.2.3"

[[claims]]
id = "C1"
status = "reproduced"
text = "It samples exactly."
reports = ["../../experiment-reports/example.md"]
```

## Notes

See [the backlog](../experiment-backlog.md).
"""

README = """# Knowledge base

| Card | Kind | Status in Thermo |
| --- | --- | --- |
| [Example library](sources/example-lib.md) | Library | `pinned_dependency` (1.2.3) |
"""


def _write_repo(root: Path, card: str = CARD, readme: str = README) -> Path:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["example-lib==1.2.3"]\n', encoding="utf-8"
    )
    (root / "uv.lock").write_text(
        '[[package]]\nname = "example-lib"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    knowledge = root / "docs/knowledge"
    (knowledge / "sources").mkdir(parents=True)
    (knowledge / "README.md").write_text(readme, encoding="utf-8")
    (knowledge / "experiment-backlog.md").write_text("# Backlog\n", encoding="utf-8")
    (knowledge / "sources/example-lib.md").write_text(card, encoding="utf-8")
    (root / "docs/experiment-reports").mkdir()
    (root / "docs/experiment-reports/example.md").write_text("# Report\n", encoding="utf-8")
    return root


def test_repository_knowledge_base_is_valid() -> None:
    assert check_knowledge_base(ROOT) == []
    assert load_cards(ROOT)


def test_minimal_repository_is_valid(tmp_path: Path) -> None:
    assert check_knowledge_base(_write_repo(tmp_path), today=TODAY) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("```toml\n", "", "must start with"),
        ('id = "example-lib"', 'id = "other-lib"', "differs from file name"),
        ('title = "Example library"', 'title = "Renamed"', "differs from heading"),
        (
            'status = "pinned_dependency"\nlast',
            'status = "verified"\nlast',
            "status: Input should be",
        ),
        ('runtime = "JAX"\n', "", "license and runtime"),
        ('reports = ["../../experiment-reports/example.md"]\n', "", "cites no report"),
        ("example.md", "missing.md", "cited report does not exist"),
        ('version = "1.2.3"\nlicense', 'version = "1.2.3"\nlicence', "Extra inputs"),
        (
            'package = "example-lib"\nversion = "1.2.3"',
            'package = "example-lib"\nversion = "9"',
            "not declared in pyproject.toml",
        ),
        ("last_checked = 2026-09-29", "last_checked = 2026-12-01", "in the future"),
        ('id = "C1"', 'id = "claim-1"', "claims.0.id: String should match"),
        ("../experiment-backlog.md", "../missing.md", "broken link"),
    ],
)
def test_card_defects_are_reported(tmp_path: Path, old: str, new: str, expected: str) -> None:
    assert old in CARD
    root = _write_repo(tmp_path, card=CARD.replace(old, new, 1))
    issues = check_knowledge_base(root, today=TODAY)
    assert any(expected in issue for issue in issues), issues


def test_arxiv_sources_must_link_their_version(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text(
        '# Paper\n\n```toml\nid = "paper"\ntitle = "Paper"\nkind = "paper"\n'
        'status = "asserted"\nlast_checked = 2026-09-29\n\n[source]\n'
        'url = "https://arxiv.org/abs/2501.00001"\nversion = "v2"\n\n'
        '[[claims]]\nid = "C1"\nstatus = "asserted"\ntext = "A claim."\n```\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact version"):
        parse_card(path)


@pytest.mark.parametrize(
    ("readme", "expected"),
    [
        (README.replace("`pinned_dependency` (1.2.3)", "`asserted`"), "must start with"),
        ("# Knowledge base\n", "missing example-lib"),
        (README + "| [Gone](sources/gone.md) | Paper | `asserted` |\n", "no valid card"),
    ],
)
def test_index_defects_are_reported(tmp_path: Path, readme: str, expected: str) -> None:
    issues = check_knowledge_base(_write_repo(tmp_path, readme=readme), today=TODAY)
    assert any(expected in issue for issue in issues), issues


def test_stale_cards_lists_only_old_cards(tmp_path: Path) -> None:
    cards = load_cards(_write_repo(tmp_path))
    assert stale_cards(cards, today=TODAY, max_age_days=30) == []
    assert stale_cards(cards, today=date(2027, 1, 1), max_age_days=30) == [
        "example-lib: last checked 2026-09-29 (94 days ago)"
    ]


def test_check_upstream_reports_newer_versions_and_failures(tmp_path: Path) -> None:
    cards = load_cards(_write_repo(tmp_path))

    def newer(url: str) -> bytes:
        assert url == "https://pypi.org/pypi/example-lib/json"
        return json.dumps({"info": {"version": "1.3.0"}}).encode()

    def failing(url: str) -> bytes:
        raise OSError("offline")

    assert check_upstream(cards, fetch=newer) == [
        "example-lib: card records 1.2.3, upstream has 1.3.0"
    ]
    assert check_upstream(cards, fetch=failing) == ["example-lib: upstream check failed: offline"]
