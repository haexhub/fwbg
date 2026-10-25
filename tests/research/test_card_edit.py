import pytest

from fwbg.research.card import (
    CardError,
    Status,
    add_section_entry,
    load_card,
    next_card_id,
    replace_card,
    section_text,
    write_card,
)
from tests.research.factories import make_card, make_trial


def test_replace_card_rewrites_the_existing_file(strategies_dir):
    path = write_card(strategies_dir, make_card(), "cusum")
    replace_card(strategies_dir, make_card(status=Status.PARKED))
    assert load_card(strategies_dir, "S-0007").status is Status.PARKED
    assert [p.name for p in (strategies_dir / "strategies").iterdir()] == [path.name]


def test_replace_card_needs_an_existing_card(strategies_dir):
    with pytest.raises(CardError, match="no research card S-0007"):
        replace_card(strategies_dir, make_card())


def test_next_card_id_follows_cards_and_ledger(strategies_dir):
    assert next_card_id(strategies_dir, []) == "S-0001"
    write_card(strategies_dir, make_card(id="S-0007"), "x")
    assert next_card_id(strategies_dir, []) == "S-0008"
    # an id whose card file is gone is never reused: its trials would follow it
    assert next_card_id(strategies_dir, [make_trial(strategy="S-0011")]) == "S-0012"


def test_next_card_id_ignores_files_that_are_not_card_files(strategies_dir):
    write_card(strategies_dir, make_card(id="S-0007"), "x")
    (strategies_dir / "strategies" / "S-draft.md").write_text("draft", encoding="utf-8")
    (strategies_dir / "strategies" / "S-00099.md").write_text("typo", encoding="utf-8")
    assert next_card_id(strategies_dir, []) == "S-0008"


def test_add_section_entry_appends_inside_the_section():
    body = "## Hypothesis\n\nH.\n\n## Verdict\n\nOpen.\n\n## Reports\n\n- a\n"
    assert add_section_entry(body, "Verdict", "### 2026-10-20: parked") == (
        "## Hypothesis\n\nH.\n\n## Verdict\n\nOpen.\n\n### 2026-10-20: parked\n\n## Reports\n\n- a\n"
    )


def test_add_section_entry_creates_a_missing_section_before_an_anchor():
    body = "## Hypothesis\n\nH.\n\n## Verdict\n\nOpen.\n"
    assert add_section_entry(body, "History", "- reopened", before=("Verdict",)) == (
        "## Hypothesis\n\nH.\n\n## History\n\n- reopened\n\n## Verdict\n\nOpen.\n"
    )


def test_add_section_entry_appends_a_missing_section_at_the_end():
    assert add_section_entry("Intro.\n", "Verdict", "x") == "Intro.\n\n## Verdict\n\nx\n"
    assert add_section_entry("", "Verdict", "x") == "## Verdict\n\nx\n"


def test_add_section_entry_skips_headings_in_code_fences():
    body = "## Verdict\n\nOpen.\n\n```\n## Fake\n```\n"
    assert add_section_entry(body, "Verdict", "x") == (
        "## Verdict\n\nOpen.\n\n```\n## Fake\n```\n\nx\n"
    )


def test_add_section_entry_ignores_a_fenced_target_heading():
    body = "## Hypothesis\n\n~~~\n## Verdict\n~~~\n"
    assert add_section_entry(body, "Verdict", "x") == (
        "## Hypothesis\n\n~~~\n## Verdict\n~~~\n\n## Verdict\n\nx\n"
    )


def test_add_section_entry_refuses_a_repeated_section():
    body = "## Verdict\n\na\n\n## Verdict\n\nb\n"
    with pytest.raises(CardError, match="section 'Verdict' appears 2 times"):
        add_section_entry(body, "Verdict", "x")


def test_section_text_returns_the_section_and_ignores_fenced_headings():
    body = "## Hypothesis\n\nOne.\n\n```\n## Mechanism\n```\n\n## Mechanism\n\nTwo.\n"
    assert section_text(body, "Hypothesis") == "One.\n\n```\n## Mechanism\n```"
    assert section_text(body, "Mechanism") == "Two."
    assert section_text(body, "Verdict") is None


def test_section_text_recognizes_long_fences():
    body = "## Hypothesis\n\nOne.\n\n````python\n## Mechanism\n````\n\n## Mechanism\n\nTwo.\n"
    assert section_text(body, "Hypothesis") == "One.\n\n````python\n## Mechanism\n````"
    assert section_text(body, "Mechanism") == "Two."
