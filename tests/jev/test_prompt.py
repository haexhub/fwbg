import pandas as pd

from scripts.jev.prompt import build_questions, build_state_text


def test_build_state_text_includes_ohlc_and_indicators():
    """State text includes OHLC prices and available indicator values."""
    row = pd.Series(
        {
            "O": 1.0845,
            "H": 1.0851,
            "L": 1.0840,
            "C": 1.0849,
            "mom_rsi_14": 61.2,
            "trend_ema_21": 1.0830,
        }
    )
    text = build_state_text(row)
    assert "1.0849" in text
    assert "mom_rsi_14=61.2" in text
    assert "trend_ema_21=1.083" in text


def test_build_state_text_skips_nan_indicators():
    """Missing indicator values are omitted from state text."""
    row = pd.Series({"O": 1.0, "H": 1.0, "L": 1.0, "C": 1.0, "mom_rsi_14": float("nan")})
    text = build_state_text(row)
    assert "mom_rsi_14" not in text


def test_build_questions_shape():
    """Long and short questions carry the requested barriers and horizon."""
    questions = build_questions(tp_pips=20, sl_pips=20, horizon_bars=30)
    assert set(questions) == {"is_long_win", "is_short_win"}
    assert questions["is_long_win"]["type"] == "noul"
    assert "20" in questions["is_long_win"]["instructions"]
    assert "30" in questions["is_long_win"]["instructions"]
