import pandas as pd
import pytest

from scripts.jev_batch_query import build_question, triple_barrier_labels


def test_triple_barrier_labels_uses_the_first_touched_barrier():
    frame = pd.DataFrame(
        {
            "H": [1.0, 1.12, 1.0],
            "L": [1.0, 0.98, 1.0],
            "C": [1.0, 1.0, 1.0],
            "_atr": [0.1, 0.1, 0.1],
        }
    )

    assert triple_barrier_labels(frame, atr_mult=1.0, timeout_bars=1) == [
        "up_first",
        "neither",
        None,
    ]

    collision = frame.copy()
    collision.loc[1, "L"] = 0.88
    assert (
        triple_barrier_labels(collision, atr_mult=1.0, timeout_bars=1)[0]
        == "down_first"
    )


def test_build_question_can_reverse_criteria_order_without_changing_values():
    normal = build_question(1.1000, 0.0010, 0.0020, horizon_bars=8)
    reversed_order = build_question(
        1.1000, 0.0010, 0.0020, horizon_bars=8, reverse_order=True
    )

    normal_criteria = normal["barrier_outcome"]["criteria"]
    reversed_criteria = reversed_order["barrier_outcome"]["criteria"]
    assert list(normal_criteria) == ["up_first", "down_first", "neither"]
    assert list(reversed_criteria) == ["down_first", "up_first", "neither"]
    assert reversed_criteria["up_first"] == normal_criteria["up_first"]


def test_triple_barrier_labels_requires_the_atr_column():
    frame = pd.DataFrame({"H": [1.0], "L": [1.0], "C": [1.0]})

    with pytest.raises(KeyError):
        triple_barrier_labels(frame, atr_mult=1.0, timeout_bars=1)
