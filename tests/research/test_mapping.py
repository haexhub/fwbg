from datetime import date, datetime

import pytest
import yaml

from fwbg.research.mapping import BackfillError, draft_mapping, load_mapping, parse_mapping
from fwbg.research.legacy import scan_runs
from tests.research.factories import STUDY, make_run


def test_draft_proposes_one_parked_card_per_name_in_order_of_first_run(runs):
    """Group runs by strategy name and number parked cards by their first run date."""
    cards = draft_mapping(runs)["cards"]
    assert [(c["id"], c["title"], c["runs"]) for c in cards] == [
        ("S-0001", "pdh retest", ["r3"]),
        ("S-0002", "orb-test", ["r1", "r2"]),
    ]
    assert {c["status"] for c in cards} == {"parked"}
    assert cards[0]["slug"] == "pdh-retest"


def test_draft_gives_a_name_without_ascii_words_a_numbered_slug(tmp_path):
    """Use a valid numbered fallback slug for a name without ASCII words."""
    results = tmp_path / "results"
    make_run(results, "r1", name="pdh retest", timestamp="2026-02-22T08:00:00")
    make_run(results, "r2", name="\u65e5\u672c", timestamp="2026-03-05T10:00:00")
    cards = draft_mapping(scan_runs(results, tmp_path / "presets"))["cards"]
    assert [(c["title"], c["slug"]) for c in cards] == [
        ("pdh retest", "pdh-retest"),
        ("\u65e5\u672c", "card-0002"),
    ]
    assert [plan.slug for plan in parse_mapping({"cards": cards})] == ["pdh-retest", "card-0002"]


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"id": "X-1"}, "invalid id"),
        ({"status": "active"}, "parked or rejected"),
        ({"slug": "Bad Slug"}, "invalid slug"),
        ({"runs": [], "studies": []}, "at least one run"),
        ({"criteria": {"min_sharpe": "high"}}, "expected a number"),
        ({"criteria": {"min_sharp": 1}}, r"S-0001\.criteria: .*unknown criterion 'min_sharp'"),
        ({"extra": 1}, "unknown keys: extra"),
        ({"studies": [{"variant": "v", "symbols": ["A"], "date": "soon", "note": "n"}]}, "date"),
        (
            {
                "studies": [
                    {"variant": "v", "symbols": ["A"], "date": date(2026, 1, 1), "note": "a\nb"}
                ]
            },
            "one line",
        ),
        (
            {
                "studies": [
                    {
                        "variant": "v",
                        "symbols": ["A"],
                        "date": date(2026, 1, 1),
                        "note": "n",
                        "metrics": {"A": {"x": float("nan")}},
                    }
                ]
            },
            "finite",
        ),
        ({"criteria": {"min_sharpe": float("inf")}}, "finite"),
        ({"criteria": {"min_trades": 10**400}}, "out of range"),
        ({"studies": [{**STUDY, "date": datetime(2026, 2, 1, 10)}]}, r"studies\[1\]\.date"),
        ({"studies": [{**STUDY, "note": "   "}]}, "one line"),
        ({"studies": [{**STUDY, "note": "trailing break\n"}]}, "one line"),
        ({"studies": [{**STUDY, "metrics": {"a": {"x": 1}, "A": {"y": 2}}}]}, "ignoring case"),
        ({"tags": ["ok", 3]}, r"S-0001\.tags: .*\['ok', 3\]"),
        ({"studies": [STUDY, {**STUDY, "note": "copy"}]}, "duplicate study variants: config-4"),
        ({"title": "!!!", "slug": None}, "S-0001: title '!!!' gives an empty slug"),
    ],
    ids=[
        "id",
        "status",
        "slug",
        "empty",
        "criteria",
        "unknown-criterion",
        "unknown",
        "date",
        "note",
        "nan",
        "inf-criterion",
        "huge-criterion",
        "datetime",
        "blank-note",
        "note-break",
        "symbol-case",
        "tags",
        "variant-twice",
        "unusable-title",
    ],
)
def test_parse_mapping_rejects_invalid_cards(runs, change, fragment):
    """Reject malformed card fields, criteria and study data with useful errors."""
    data = draft_mapping(runs)
    data["cards"][0].update(change)
    with pytest.raises(BackfillError, match=fragment):
        parse_mapping(data)


def test_duplicate_card_ids_are_rejected(runs):
    """Reject two mapping entries that share a research card ID."""
    data = draft_mapping(runs)
    data["cards"][1]["id"] = "S-0001"
    with pytest.raises(BackfillError, match="duplicate card ids: S-0001"):
        parse_mapping(data)


def test_load_mapping_reads_a_yaml_draft(runs):
    """Load a serialised YAML draft into the same plans as its original mapping."""
    text = yaml.safe_dump(draft_mapping(runs), sort_keys=False, allow_unicode=True)
    assert load_mapping(text) == parse_mapping(draft_mapping(runs))


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (
            "cards:\n"
            "- id: S-0001\n"
            "  title: jev\n"
            "  studies:\n"
            "  - {variant: a, symbols: [EURUSD], date: 2026-09-25, note: first}\n"
            "  studies:\n"
            "  - {variant: b, symbols: [EURUSD], date: 2026-09-26, note: second}\n",
            "line 6, column 3: duplicate key 'studies'",
        ),
        (
            "cards:\n- {id: S-0001, title: x, studies: [{date: 2026-02-30}]}\n",
            # Python 3.14 reworded the date error ("day 30 must be in range 1..28 ...").
            r"not valid YAML: day .*range",
        ),
        ("cards: [\n", "line 2, column 1"),
    ],
    ids=["duplicate-key", "impossible-date", "syntax"],
)
def test_load_mapping_reports_yaml_errors(text, fragment):
    """Report duplicate keys, impossible dates and YAML syntax errors."""
    with pytest.raises(BackfillError, match=fragment):
        load_mapping(text)


def test_unknown_top_level_keys_are_refused():
    # A dedented block lands at the top level instead of in its card.
    """Catch accidentally dedented mapping fields as unknown top-level keys."""
    text = (
        "cards:\n"
        "- id: S-0001\n"
        "  title: jev\n"
        "  runs: [r1]\n"
        "studies:\n"
        "- {variant: a, symbols: [EURUSD], date: 2026-09-25, note: first}\n"
    )
    with pytest.raises(BackfillError, match="unknown top-level keys: studies"):
        load_mapping(text)
