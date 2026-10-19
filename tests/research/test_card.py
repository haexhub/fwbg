from datetime import date
from pathlib import Path

import pytest

from fwbg.research.card import CardError, Status, criteria_hash, load_card, parse_card

REGISTERED_HASH = "sha256:5272a26ad425169031b39ae95c05abe1e2c3ebb664faaae0508eef108b989970"
AMENDED_HASH = "sha256:c22558e72bca2f92a641914684ced8140937d9fe44d872db8f2526a71139f7e2"

AMENDMENTS = f"""\
criteria_amendments:
  - date: 2026-10-20
    reason: daily needs fewer trades
    criteria_hash: {AMENDED_HASH}
"""

CARD = f"""\
---
id: S-0007
title: CUSUM event continuation
status: active
tags: [event, daily]
created: 2026-10-19
holdout_cutoff: "2025-04-30"
holdout_contaminated: false
success_criteria:
  min_sharpe: 1.0
  min_trades: 100
registered_criteria_hash: {REGISTERED_HASH}
{AMENDMENTS}---
## Hypothesis
Structural breaks continue.
"""


def test_parses_front_matter_and_body():
    card = parse_card(CARD)
    assert card.id == "S-0007"
    assert card.status is Status.ACTIVE
    assert card.tags == ("event", "daily")
    assert card.created == date(2026, 10, 19)
    assert card.holdout_cutoff == date(2025, 4, 30)
    assert card.holdout_contaminated is False
    assert card.success_criteria == {"min_sharpe": 1.0, "min_trades": 100.0}
    assert card.registered_criteria_hash == (
        "sha256:5272a26ad425169031b39ae95c05abe1e2c3ebb664faaae0508eef108b989970"
    )
    assert card.criteria_amendments[0].criteria_hash == (
        "sha256:c22558e72bca2f92a641914684ced8140937d9fe44d872db8f2526a71139f7e2"
    )
    assert card.body.startswith("## Hypothesis")


def test_parses_registration_card_without_tags_or_amendments():
    card = parse_card(CARD.replace("tags: [event, daily]\n", "").replace(AMENDMENTS, ""))
    assert card.tags == ()
    assert card.criteria_amendments == ()


def test_parses_contaminated_holdout():
    card = parse_card(CARD.replace("holdout_contaminated: false", "holdout_contaminated: true"))
    assert card.holdout_contaminated is True


def test_datetime_becomes_date():
    card = parse_card(CARD.replace("created: 2026-10-19", "created: 2026-10-19 10:00:00"))
    assert type(card.created) is date
    assert card.created == date(2026, 10, 19)


def test_opening_fence_may_have_trailing_whitespace():
    assert parse_card(CARD.replace("---\n", "--- \n", 1)).id == "S-0007"


def test_success_criteria_are_immutable():
    card = parse_card(CARD)
    with pytest.raises(TypeError):
        card.success_criteria["min_sharpe"] = 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("broken", "fragment"),
    [
        (CARD.replace("---\nid", "id", 1), "front matter"),
        (CARD.replace("title: CUSUM event continuation\n", ""), "title"),
        (CARD.replace("title: CUSUM event continuation", "title:"), "title"),
        (CARD.replace("status: active", "status: maybe"), "maybe"),
        (
            CARD.replace("status: active", "status: active\nstatus: rejected"),
            "duplicate key 'status'",
        ),
        (CARD.replace('holdout_cutoff: "2025-04-30"', "holdout_cutoff: someday"), "holdout_cutoff"),
        (
            CARD.replace('holdout_cutoff: "2025-04-30"', "holdout_cutoff: 2025-04-31"),
            "invalid YAML",
        ),
        (CARD.replace("holdout_contaminated: false\n", ""), "holdout_contaminated"),
        (
            CARD.replace("holdout_contaminated: false", 'holdout_contaminated: "no"'),
            "holdout_contaminated",
        ),
        (
            CARD.replace("holdout_contaminated: false", "holdout_contaminted: true"),
            "holdout_contaminted",
        ),
        (CARD.replace("  min_sharpe: 1.0\n  min_trades: 100\n", ""), "success_criteria"),
        (
            CARD.replace("  min_sharpe: 1.0\n  min_trades: 100\n", "").replace(
                "success_criteria:", "success_criteria: {}"
            ),
            "success_criteria",
        ),
        (CARD.replace("min_trades: 100", "min_trades: yes"), "min_trades"),
        (CARD.replace("min_sharpe: 1.0", "min_sharpe: .nan"), "min_sharpe"),
        (CARD.replace("tags: [event, daily]", "tags: event"), "tags"),
        (CARD.replace("id: S-0007", "id: 7"), "invalid research id"),
        (CARD.replace("id: S-0007", "id: S-\u0660\u0660\u0660\u0667"), "invalid research id"),
        (CARD.replace(REGISTERED_HASH, "123"), "registered_criteria_hash"),
        (CARD.replace(AMENDMENTS, "criteria_amendments: 5\n"), "criteria_amendments"),
        (
            CARD.replace(AMENDMENTS, "criteria_amendments:\n  date: 2026-10-20\n  reason: x\n"),
            "criteria_amendments",
        ),
        (CARD.replace("    reason: daily needs fewer trades\n", ""), "reason"),
        (CARD.replace("reason: daily needs fewer trades", 'reason: ""'), "reason"),
        (CARD.replace(AMENDED_HASH, "sha256:abc"), "criteria_amendments.criteria_hash"),
    ],
    ids=[
        "no-front-matter",
        "missing-title",
        "null-title",
        "unknown-status",
        "duplicate-key",
        "non-date-cutoff",
        "impossible-date",
        "missing-contaminated",
        "string-contaminated",
        "unknown-key",
        "null-criteria",
        "empty-criteria",
        "bool-criterion",
        "nan-criterion",
        "scalar-tags",
        "numeric-id",
        "non-ascii-digit-id",
        "numeric-registered-hash",
        "scalar-amendments",
        "mapping-amendments",
        "amendment-without-reason",
        "amendment-empty-reason",
        "amendment-short-hash",
    ],
)
def test_rejects_malformed_cards(broken, fragment):
    with pytest.raises(CardError, match=fragment):
        parse_card(broken)


def test_registered_hash_is_digest_of_criteria():
    card = parse_card(CARD)
    assert card.criteria_hash == card.registered_criteria_hash


def test_criteria_hash_ignores_key_order_but_not_values():
    assert criteria_hash({"a": 1.0, "b": 2.0}) == criteria_hash({"b": 2.0, "a": 1.0})
    assert criteria_hash({"a": 1.0}) != criteria_hash({"a": 1.5})
    assert criteria_hash({"a": 1.0, "b": 2.0}) != criteria_hash({"a": 2.0, "b": 1.0})


def write_card(research_dir: Path, filename: str, text: str = CARD) -> None:
    folder = research_dir / "strategies"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(text, encoding="utf-8")


def test_load_card_by_id(tmp_path):
    write_card(tmp_path, "S-0007-cusum-continuation.md")
    assert load_card(tmp_path, "S-0007").title == "CUSUM event continuation"


def test_load_card_without_slug(tmp_path):
    write_card(tmp_path, "S-0007.md")
    assert load_card(tmp_path, "S-0007").id == "S-0007"


def test_load_card_tolerates_byte_order_mark(tmp_path):
    write_card(tmp_path, "S-0007.md", "\ufeff" + CARD)
    assert load_card(tmp_path, "S-0007").id == "S-0007"


def test_load_card_returns_none_when_absent(tmp_path):
    write_card(tmp_path, "S-0007-cusum-continuation.md")
    assert load_card(tmp_path, "S-0008") is None


def test_load_card_does_not_match_longer_ids(tmp_path):
    write_card(tmp_path, "S-00071-other.md")
    assert load_card(tmp_path, "S-0007") is None


def test_load_card_refuses_ambiguous_ids(tmp_path):
    write_card(tmp_path, "S-0007-a.md")
    write_card(tmp_path, "S-0007-b.md")
    with pytest.raises(CardError, match="ambiguous"):
        load_card(tmp_path, "S-0007")


def test_load_card_refuses_bare_and_slugged_file_for_same_id(tmp_path):
    write_card(tmp_path, "S-0007.md")
    write_card(tmp_path, "S-0007-x.md")
    with pytest.raises(CardError, match="ambiguous"):
        load_card(tmp_path, "S-0007")


def test_load_card_refuses_mismatched_id(tmp_path):
    write_card(tmp_path, "S-0008-copy.md")
    with pytest.raises(CardError, match="declares id S-0007"):
        load_card(tmp_path, "S-0008")


def test_load_card_names_file_of_malformed_card(tmp_path):
    write_card(tmp_path, "S-0007-broken.md", CARD.replace("status: active", "status: maybe"))
    with pytest.raises(CardError, match=r"^S-0007-broken\.md: "):
        load_card(tmp_path, "S-0007")


def test_load_card_refuses_malformed_id(tmp_path):
    with pytest.raises(CardError, match="invalid research id"):
        load_card(tmp_path, "../S-0007")
