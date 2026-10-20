import errno
from datetime import date, datetime
from pathlib import Path

import pytest

from fwbg.research.card import (
    CardError,
    Status,
    criteria_hash,
    load_card,
    parse_card,
    render_card,
    slugify,
    write_card,
)
from tests.research.factories import make_card

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
    assert card.criteria_amendments[0].date == date(2026, 10, 20)
    assert card.criteria_amendments[0].reason == "daily needs fewer trades"
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
        (CARD.replace("title: CUSUM event continuation", 'title: "   "'), "title"),
        (CARD.replace("status: active", "status: maybe"), "status.*maybe.*active"),
        (
            CARD.replace("status: active", "status: active\nstatus: rejected"),
            "duplicate key 'status'",
        ),
        (
            CARD.replace("  min_sharpe: 1.0\n", "  min_sharpe: 1.0\n  min_sharpe: 2.0\n"),
            "duplicate key 'min_sharpe'",
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
        (
            CARD.replace("min_trades: 100", "min_trades: 1" + "0" * 400),
            "min_trades.*out of range",
        ),
        (CARD.replace("min_trades: 100", "1: 5"), "success_criteria.*key"),
        (CARD.replace("tags: [event, daily]", "tags: event"), "tags"),
        (CARD.replace("tags: [event, daily]", "tags: [2026]"), "tags"),
        (CARD.replace("id: S-0007", "id: 7"), "invalid research id"),
        (CARD.replace("id: S-0007", "id: S-\u0660\u0660\u0660\u0667"), "invalid research id"),
        (CARD.replace(REGISTERED_HASH, "123"), "registered_criteria_hash"),
        (CARD.replace(AMENDMENTS, "criteria_amendments: 5\n"), "criteria_amendments"),
        (
            CARD.replace(AMENDMENTS, "criteria_amendments:\n  date: 2026-10-20\n  reason: x\n"),
            "criteria_amendments",
        ),
        (
            CARD.replace(AMENDMENTS, "criteria_amendments:\n  - foo\n"),
            "criteria_amendments.*'foo'",
        ),
        (
            CARD.replace(
                "    reason: daily needs fewer trades\n",
                "    reason: daily needs fewer trades\n    note: x\n",
            ),
            "criteria_amendments.*note",
        ),
        (CARD.replace("    reason: daily needs fewer trades\n", ""), "reason"),
        (CARD.replace("reason: daily needs fewer trades", 'reason: ""'), "reason"),
        (CARD.replace(AMENDED_HASH, "sha256:abc"), "criteria_amendments.criteria_hash"),
    ],
    ids=[
        "no-front-matter",
        "missing-title",
        "null-title",
        "blank-title",
        "unknown-status",
        "duplicate-key",
        "nested-duplicate-key",
        "non-date-cutoff",
        "impossible-date",
        "missing-contaminated",
        "string-contaminated",
        "unknown-key",
        "null-criteria",
        "empty-criteria",
        "bool-criterion",
        "nan-criterion",
        "overflowing-criterion",
        "non-string-criterion-key",
        "scalar-tags",
        "non-string-tag",
        "numeric-id",
        "non-ascii-digit-id",
        "numeric-registered-hash",
        "scalar-amendments",
        "mapping-amendments",
        "scalar-amendment-item",
        "amendment-unknown-key",
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


def write_card_file(research_dir: Path, filename: str, text: str = CARD) -> None:
    """Write a card fixture under the register strategies directory."""
    folder = research_dir / "strategies"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(text, encoding="utf-8")


def test_load_card_by_id(tmp_path):
    """Load a slugged card file by its research ID."""
    write_card_file(tmp_path, "S-0007-cusum-continuation.md")
    assert load_card(tmp_path, "S-0007").title == "CUSUM event continuation"


def test_load_card_without_slug(tmp_path):
    """Load a card whose filename contains only its research ID."""
    write_card_file(tmp_path, "S-0007.md")
    assert load_card(tmp_path, "S-0007").id == "S-0007"


def test_load_card_tolerates_byte_order_mark(tmp_path):
    """Accept a UTF-8 byte order mark before card front matter."""
    write_card_file(tmp_path, "S-0007.md", "\ufeff" + CARD)
    assert load_card(tmp_path, "S-0007").id == "S-0007"


def test_load_card_returns_none_when_absent(tmp_path):
    """Return None when no card file matches the requested ID."""
    write_card_file(tmp_path, "S-0007-cusum-continuation.md")
    assert load_card(tmp_path, "S-0008") is None


def test_load_card_does_not_match_longer_ids(tmp_path):
    """Keep a research ID from matching a longer ID with the same prefix."""
    write_card_file(tmp_path, "S-00071-other.md")
    assert load_card(tmp_path, "S-0007") is None


def test_load_card_refuses_ambiguous_ids(tmp_path):
    """Reject multiple slugged card files for the same research ID."""
    write_card_file(tmp_path, "S-0007-a.md")
    write_card_file(tmp_path, "S-0007-b.md")
    with pytest.raises(CardError, match="ambiguous"):
        load_card(tmp_path, "S-0007")


def test_load_card_refuses_bare_and_slugged_file_for_same_id(tmp_path):
    """Reject a bare card filename alongside a slugged file for the same ID."""
    write_card_file(tmp_path, "S-0007.md")
    write_card_file(tmp_path, "S-0007-x.md")
    with pytest.raises(CardError, match="ambiguous"):
        load_card(tmp_path, "S-0007")


def test_load_card_refuses_mismatched_id(tmp_path):
    """Reject a card whose declared ID differs from the requested filename ID."""
    write_card_file(tmp_path, "S-0008-copy.md")
    with pytest.raises(CardError, match="declares id S-0007"):
        load_card(tmp_path, "S-0008")


def test_load_card_names_file_of_malformed_card(tmp_path):
    """Include the card filename when reporting malformed front matter."""
    write_card_file(tmp_path, "S-0007-broken.md", CARD.replace("status: active", "status: maybe"))
    with pytest.raises(CardError, match=r"^S-0007-broken\.md: "):
        load_card(tmp_path, "S-0007")


def test_load_card_refuses_non_utf8_file(tmp_path):
    folder = tmp_path / "strategies"
    folder.mkdir()
    (folder / "S-0007-x.md").write_bytes(b"\xff\xfe---")
    with pytest.raises(CardError, match=r"^S-0007-x\.md: not valid UTF-8"):
        load_card(tmp_path, "S-0007")


def test_load_card_refuses_malformed_id(tmp_path):
    with pytest.raises(CardError, match="invalid research id"):
        load_card(tmp_path, "../S-0007")


def test_render_round_trips_through_parse():
    """Preserve all card fields when rendering and parsing an amended card."""
    card = parse_card(CARD)
    assert parse_card(render_card(card)) == card


def test_render_round_trips_awkward_titles_and_no_amendments():
    """Preserve punctuation and Unicode in a card with no tags or amendments."""
    card = make_card(title='ORB: 15m ä "quoted" #1', tags=(), body="")
    assert parse_card(render_card(card)) == card


def test_render_writes_the_committed_card_layout():
    """Verify the exact Markdown and YAML layout written for a card."""
    card = make_card(title="ORB ä", tags=("a",), body="## Hypothesis\n")
    assert render_card(card) == (
        "---\n"
        "id: S-0007\n"
        "title: ORB ä\n"
        "status: active\n"
        "tags:\n"
        "- a\n"
        "created: 2026-10-19\n"
        "holdout_cutoff: 2025-04-30\n"
        "holdout_contaminated: false\n"
        "success_criteria:\n"
        "  min_sharpe: 1.0\n"
        "  min_trades: 100.0\n"
        f"registered_criteria_hash: {REGISTERED_HASH}\n"
        "---\n"
        "## Hypothesis\n"
    )


def test_render_keeps_long_titles_on_one_line():
    """Prevent YAML wrapping from splitting long card titles."""
    title = " ".join(["word"] * 40)
    assert f"\ntitle: {title}\n" in render_card(make_card(title=title))


def test_render_refuses_a_card_that_would_not_read_back():
    """Reject a card whose datetime would change when parsed back as a date."""
    with pytest.raises(CardError, match="would not read back"):
        render_card(make_card(created=datetime(2026, 10, 19, 10, 0)))


def test_render_names_the_card_when_it_would_not_read_back():
    """Identify the card and criterion when rendering rejects nonfinite criteria."""
    with pytest.raises(
        CardError, match=r"^S-0007 would not read back: success_criteria\.min_sharpe"
    ):
        render_card(make_card(success_criteria={"min_sharpe": float("inf")}))


def test_render_refuses_carriage_returns_in_the_body():
    """Reject prose whose carriage returns would change during a file round trip."""
    with pytest.raises(CardError, match="carriage return"):
        render_card(make_card(body="## Hypothesis\r\nnotes\r\n"))


def test_write_card_writes_once_and_never_overwrites(tmp_path):
    """Write a readable card once and refuse reuse of its ID with another slug."""
    card = make_card()
    (tmp_path / "research" / "strategies").mkdir(parents=True)
    path = write_card(tmp_path / "research", card, "cusum-continuation")
    assert path.name == "S-0007-cusum-continuation.md"
    assert load_card(tmp_path / "research", "S-0007") == card
    with pytest.raises(CardError, match="never overwritten"):
        write_card(tmp_path / "research", card, "other-slug")


def test_write_card_never_overwrites_an_existing_file(tmp_path, monkeypatch):
    """Protect an existing file even when the preliminary card lookup misses it."""
    monkeypatch.setattr("fwbg.research.card.load_card", lambda *_: None)
    target = tmp_path / "strategies" / "S-0007-cusum-continuation.md"
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    with pytest.raises(CardError, match="never overwritten"):
        write_card(tmp_path, make_card(), "cusum-continuation")
    assert target.read_text(encoding="utf-8") == "x"


def test_write_card_removes_a_partial_file_after_a_failed_write(tmp_path, monkeypatch):
    """Remove partial card bytes after a simulated disk-full write failure."""
    real_open = Path.open

    class DiskFull:
        """Writes a few bytes, then fails like a full disk."""

        def __init__(self, handle):
            """Keep the underlying file handle for the simulated disk failure."""
            self.handle = handle

        def __enter__(self):
            """Return the failing writer when entering its file context."""
            return self

        def __exit__(self, *_):
            """Close the underlying file handle when leaving the context."""
            self.handle.close()

        def write(self, text):
            """Flush a partial write before raising a disk-full error."""
            self.handle.write(text[:4])
            self.handle.flush()
            raise OSError(errno.ENOSPC, "No space left on device")

    def open_disk_full(path, mode="r", *args, **kwargs):
        """Wrap exclusively created files with a writer that simulates disk exhaustion."""
        handle = real_open(path, mode, *args, **kwargs)
        return DiskFull(handle) if mode == "x" else handle

    monkeypatch.setattr(Path, "open", open_disk_full)
    (tmp_path / "strategies").mkdir()
    with pytest.raises(OSError, match="No space left"):
        write_card(tmp_path, make_card(), "cusum-continuation")
    assert list((tmp_path / "strategies").iterdir()) == []


@pytest.mark.parametrize(
    "slug",
    ["", "-a", "a-", "a--b", "a_b", "a/b", "../x", "a.b", "a" * 61, "Not A Slug"],
    ids=[
        "empty",
        "leading-hyphen",
        "trailing-hyphen",
        "double-hyphen",
        "underscore",
        "slash",
        "parent-dir",
        "dot",
        "61-chars",
        "spaces-and-capitals",
    ],
)
def test_write_card_rejects_invalid_slug(tmp_path, slug):
    """Reject unsafe or noncanonical slugs before creating a card file."""
    (tmp_path / "strategies").mkdir()
    with pytest.raises(CardError, match="invalid slug"):
        write_card(tmp_path, make_card(), slug)


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("FWBG creative CUSUM event continuation", "fwbg-creative-cusum-event-continuation"),
        ("ORB + PDH/PDL Scalping", "orb-pdh-pdl-scalping"),
        ("m3_smoke_20260623", "m3-smoke-20260623"),
    ],
)
def test_slugify(title, slug):
    """Convert representative strategy titles into stable lowercase ASCII slugs."""
    assert slugify(title) == slug


def test_slugify_caps_at_60_characters_without_a_trailing_hyphen():
    """Truncate slugs to sixty characters and strip a trailing separator."""
    assert slugify("a" * 59 + " bc") == "a" * 59


def test_slugify_refuses_titles_without_ascii_words():
    """Reject titles that cannot produce a nonempty ASCII slug."""
    with pytest.raises(CardError, match="empty slug"):
        slugify("äöü")


def test_unknown_criterion_is_refused():
    text = render_card(make_card()).replace("min_trades:", "min_trade:")
    with pytest.raises(CardError, match="unknown criterion 'min_trade'"):
        parse_card(text)
