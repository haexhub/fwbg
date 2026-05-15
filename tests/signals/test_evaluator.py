"""Tests for the signal rules evaluator."""

import numpy as np
import pandas as pd
import pytest

from fwbg.signals.evaluator import (
    evaluate_condition,
    evaluate_rules,
    resolve_column,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def df():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    return pd.DataFrame(
        {
            "rb1_orb_s08_breakout_up": [0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
            "rb1_orb_s08_breakout_down": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            "adx_14": [10, 15, 25, 30, 35, 20, 18, 40, 22, 28],
            "mom_rsi_14": [50, 55, 70, 80, 30, 45, 60, 25, 35, 65],
            "ema_9": [100, 101, 103, 105, 104, 102, 103, 106, 107, 108],
            "ema_21": [100, 100, 101, 102, 103, 103, 104, 104, 105, 106],
            "close": [100, 102, 104, 106, 103, 101, 103, 107, 108, 110],
        },
        index=idx,
    )


# ===================================================================
# TestResolveColumn
# ===================================================================

class TestResolveColumn:
    def test_exact_match(self):
        cols = ["close", "ema_9", "ema_21"]
        assert resolve_column("close", cols) == "close"
        assert resolve_column("ema_9", cols) == "ema_9"

    def test_suffix_match(self):
        cols = ["rb1_orb_s08_breakout_up", "rb1_orb_s08_breakout_down", "adx_14"]
        assert resolve_column("breakout_up", cols) == "rb1_orb_s08_breakout_up"
        assert resolve_column("adx_14", cols) == "adx_14"

    def test_not_found_raises(self):
        cols = ["close", "ema_9"]
        with pytest.raises(KeyError, match="not found"):
            resolve_column("nonexistent", cols)

    def test_ambiguous_raises(self):
        cols = ["fast_ema_9", "slow_ema_9"]
        with pytest.raises(KeyError, match="Ambiguous"):
            resolve_column("ema_9", cols)


# ===================================================================
# TestSignalActive
# ===================================================================

class TestSignalActive:
    def test_basic(self, df):
        cond = {"type": "signal_active", "column": "breakout_up"}
        result = evaluate_condition(cond, df)
        expected = pd.Series(
            [False, False, True, True, False, False, True, False, False, True],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nan_handling(self, df):
        """NaN values should be treated as not active (0)."""
        df_copy = df.copy()
        df_copy.loc[df_copy.index[2], "rb1_orb_s08_breakout_up"] = np.nan
        cond = {"type": "signal_active", "column": "breakout_up"}
        result = evaluate_condition(cond, df_copy)
        # index 2 was 1 but is now NaN → should be False
        assert result.iloc[2] is np.bool_(False)


# ===================================================================
# TestValueCheck
# ===================================================================

class TestValueCheck:
    def test_less_than(self, df):
        cond = {"type": "value_check", "column": "adx_14", "op": "<", "value": 20}
        result = evaluate_condition(cond, df)
        # adx: 10,15,25,30,35,20,18,40,22,28
        expected = pd.Series(
            [True, True, False, False, False, False, True, False, False, False],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_greater_equal(self, df):
        cond = {"type": "value_check", "column": "adx_14", "op": ">=", "value": 25}
        result = evaluate_condition(cond, df)
        # adx: 10,15,25,30,35,20,18,40,22,28
        expected = pd.Series(
            [False, False, True, True, True, False, False, True, False, True],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_equal(self, df):
        cond = {"type": "value_check", "column": "adx_14", "op": "==", "value": 25}
        result = evaluate_condition(cond, df)
        expected = pd.Series(
            [False, False, True, False, False, False, False, False, False, False],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nan_handling(self, df):
        df_copy = df.copy()
        df_copy.loc[df_copy.index[0], "adx_14"] = np.nan
        cond = {"type": "value_check", "column": "adx_14", "op": "<", "value": 20}
        result = evaluate_condition(cond, df_copy)
        # index 0 was 10 (True) but is now NaN → should be False
        assert result.iloc[0] is np.bool_(False)


# ===================================================================
# TestColCompare
# ===================================================================

class TestColCompare:
    def test_column_greater_than(self, df):
        cond = {"type": "col_compare", "column_a": "ema_9", "column_b": "ema_21", "op": ">"}
        result = evaluate_condition(cond, df)
        # ema_9:  100,101,103,105,104,102,103,106,107,108
        # ema_21: 100,100,101,102,103,103,104,104,105,106
        expected = pd.Series(
            [False, True, True, True, True, False, False, True, True, True],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_column_less_equal(self, df):
        cond = {"type": "col_compare", "column_a": "ema_9", "column_b": "ema_21", "op": "<="}
        result = evaluate_condition(cond, df)
        # inverse of > (since we don't have NaN here)
        expected = pd.Series(
            [True, False, False, False, False, True, True, False, False, False],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)


# ===================================================================
# TestCrossing
# ===================================================================

class TestCrossing:
    def test_crosses_above(self, df):
        cond = {
            "type": "crossing",
            "column_a": "ema_9",
            "column_b": "ema_21",
            "direction": "above",
        }
        result = evaluate_condition(cond, df)
        # ema_9:  100,101,103,105,104,102,103,106,107,108
        # ema_21: 100,100,101,102,103,103,104,104,105,106
        # a > b:  F,  T,  T,  T,  T,  F,  F,  T,  T,  T
        # a<=b prev: -, T, F, F, F, F, T, T, F, F
        # cross above: F, T, F, F, F, F, F, T, F, F
        expected = pd.Series(
            [False, True, False, False, False, False, False, True, False, False],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_crosses_below(self, df):
        cond = {
            "type": "crossing",
            "column_a": "ema_9",
            "column_b": "ema_21",
            "direction": "below",
        }
        result = evaluate_condition(cond, df)
        # a < b:  F,  F,  F,  F,  F,  T,  T,  F,  F,  F
        # a>=b prev: -, T, T, T, T, T, F, F, T, T
        # cross below: F, F, F, F, F, T, F, F, F, F
        expected = pd.Series(
            [False, False, False, False, False, True, False, False, False, False],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_invalid_direction(self, df):
        cond = {
            "type": "crossing",
            "column_a": "ema_9",
            "column_b": "ema_21",
            "direction": "sideways",
        }
        with pytest.raises(ValueError, match="Unknown crossing direction"):
            evaluate_condition(cond, df)


# ===================================================================
# TestNestedRules
# ===================================================================

class TestNestedRules:
    def test_and_group(self, df):
        """breakout_up AND adx >= 25."""
        rules = {
            "operator": "AND",
            "conditions": [
                {"type": "signal_active", "column": "breakout_up"},
                {"type": "value_check", "column": "adx_14", "op": ">=", "value": 25},
            ],
        }
        result = evaluate_rules(rules, df)
        # breakout_up: F,F,T,T,F,F,T,F,F,T
        # adx>=25:     F,F,T,T,T,F,F,T,F,T
        # AND:         F,F,T,T,F,F,F,F,F,T
        expected = pd.Series(
            [False, False, True, True, False, False, False, False, False, True],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_or_group(self, df):
        """breakout_up OR breakout_down."""
        rules = {
            "operator": "OR",
            "conditions": [
                {"type": "signal_active", "column": "breakout_up"},
                {"type": "signal_active", "column": "breakout_down"},
            ],
        }
        result = evaluate_rules(rules, df)
        # up:   F,F,T,T,F,F,T,F,F,T
        # down: F,T,F,F,T,F,F,F,T,F
        # OR:   F,T,T,T,T,F,T,F,T,T
        expected = pd.Series(
            [False, True, True, True, True, False, True, False, True, True],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nested_group(self, df):
        """AND(breakout_up, OR(adx >= 25, rsi > 60))."""
        rules = {
            "operator": "AND",
            "conditions": [
                {"type": "signal_active", "column": "breakout_up"},
                {
                    "type": "group",
                    "operator": "OR",
                    "conditions": [
                        {"type": "value_check", "column": "adx_14", "op": ">=", "value": 25},
                        {"type": "value_check", "column": "mom_rsi_14", "op": ">", "value": 60},
                    ],
                },
            ],
        }
        result = evaluate_rules(rules, df)
        # breakout_up: F,F,T,T,F,F,T,F,F,T
        # adx>=25:     F,F,T,T,T,F,F,T,F,T
        # rsi>60:      F,F,T,T,F,F,F,F,F,T
        # OR(adx,rsi): F,F,T,T,T,F,F,T,F,T
        # AND:         F,F,T,T,F,F,F,F,F,T
        expected = pd.Series(
            [False, False, True, True, False, False, False, False, False, True],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_empty_conditions(self, df):
        """Empty conditions should return all True."""
        rules = {"operator": "AND", "conditions": []}
        result = evaluate_rules(rules, df)
        expected = pd.Series(True, index=df.index, dtype=bool)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_invalid_operator(self, df):
        rules = {"operator": "XOR", "conditions": []}
        with pytest.raises(ValueError, match="Unknown operator"):
            evaluate_rules(rules, df)

    def test_invalid_condition_type(self, df):
        rules = {
            "operator": "AND",
            "conditions": [{"type": "magic", "column": "close"}],
        }
        with pytest.raises(ValueError, match="Unknown condition type"):
            evaluate_rules(rules, df)


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases:
    def test_nan_in_crossing(self, df):
        """NaN at shift boundary (first row) should be False."""
        cond = {
            "type": "crossing",
            "column_a": "ema_9",
            "column_b": "ema_21",
            "direction": "above",
        }
        result = evaluate_condition(cond, df)
        # First row has no previous row → shift produces NaN → should be False
        assert result.iloc[0] is np.bool_(False)

    def test_nan_in_col_compare(self, df):
        df_copy = df.copy()
        df_copy.loc[df_copy.index[3], "ema_9"] = np.nan
        cond = {"type": "col_compare", "column_a": "ema_9", "column_b": "ema_21", "op": ">"}
        result = evaluate_condition(cond, df_copy)
        # index 3 had ema_9=105 > ema_21=102, but NaN should → False
        assert result.iloc[3] is np.bool_(False)

    def test_all_nan_column(self, df):
        df_copy = df.copy()
        df_copy["adx_14"] = np.nan
        cond = {"type": "value_check", "column": "adx_14", "op": ">=", "value": 25}
        result = evaluate_condition(cond, df_copy)
        assert not result.any()

    def test_single_condition_and(self, df):
        """Single condition in AND group should just return that condition."""
        rules = {
            "operator": "AND",
            "conditions": [
                {"type": "signal_active", "column": "breakout_up"},
            ],
        }
        result = evaluate_rules(rules, df)
        expected = pd.Series(
            [False, False, True, True, False, False, True, False, False, True],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_default_operator_is_and(self, df):
        """If operator is not specified, default to AND."""
        rules = {
            "conditions": [
                {"type": "signal_active", "column": "breakout_up"},
                {"type": "value_check", "column": "adx_14", "op": ">=", "value": 25},
            ],
        }
        result = evaluate_rules(rules, df)
        expected = pd.Series(
            [False, False, True, True, False, False, False, False, False, True],
            index=df.index,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_invalid_op(self, df):
        cond = {"type": "value_check", "column": "close", "op": "~=", "value": 100}
        with pytest.raises(ValueError, match="Unsupported operator"):
            evaluate_condition(cond, df)
