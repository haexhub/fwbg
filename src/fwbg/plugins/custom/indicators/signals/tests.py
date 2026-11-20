"""Tests for the computed_signal plugin and its definition schema."""

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_signals = import_plugin_module("custom", "indicators", "signals")
if _signals is None:
    pytest.skip("custom signals plugin not available", allow_module_level=True)


def _frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(cols, index=idx)


def _plugin(rules):
    cls = type("Dyn", (_signals.ComputedSignalPlugin,), {"_rules": rules})
    return cls()


def _rule(op, value, output="sig", column="x"):
    return {
        "output": output,
        "logic": "AND",
        "conditions": [{"column": column, "op": op, "value": value}],
    }


class TestOutputNameValidation:
    def test_ohlcv_output_is_rejected_and_close_is_not_overwritten(self):
        df = _frame(C=[1.0, 2.0, 3.0], x=[0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="OHLCV"):
            _plugin([_rule(">", 0.5, output="C")]).compute(df)
        assert list(df["C"]) == [1.0, 2.0, 3.0]

    def test_existing_column_is_rejected(self):
        df = _frame(C=[1.0, 2.0], x=[0.0, 1.0], other=[1.0, 1.0])
        with pytest.raises(ValueError, match="already exists"):
            _plugin([_rule(">", 0.5, output="other")]).compute(df)

    def test_duplicate_outputs_within_one_definition_are_rejected(self):
        df = _frame(C=[1.0, 2.0], x=[0.0, 1.0])
        rules = [_rule(">", 0.5, output="s"), _rule("<", 0.5, output="s")]
        with pytest.raises(ValueError, match="already exists"):
            _plugin(rules).compute(df)

    @pytest.mark.parametrize("name", ["", "1abc", "a b", "a-b", "x;y", "a.b"])
    def test_invalid_identifier_is_rejected(self, name):
        with pytest.raises(ValueError, match="identifier"):
            _signals.validate_output_name(name)

    def test_valid_name_passes(self):
        assert _signals.validate_output_name("sig_1", existing=["C"]) == "sig_1"

    def test_api_schema_rejects_bad_outputs(self):
        pytest.importorskip("fastapi")
        from pydantic import ValidationError

        from fwbg.api.custom_signals import SignalDefinition

        cond = {"column": "x", "op": ">", "value": 1}

        def definition(*outputs):
            return {
                "name": "s",
                "rules": [{"output": o, "conditions": [cond]} for o in outputs],
            }

        SignalDefinition(**definition("ok_signal"))
        for bad in ("C", "not valid", "1x"):
            with pytest.raises(ValidationError):
                SignalDefinition(**definition(bad))
        with pytest.raises(ValidationError):
            SignalDefinition(**definition("a", "a"))


class TestNaNSemantics:
    def test_not_equal_is_nan_for_nan_input(self):
        df = _frame(C=[1.0] * 4, x=[np.nan, 1.0, 2.0, np.nan])
        out = _plugin([_rule("!=", 1.0)]).compute(df)["sig"]
        # shifted by one bar: row i holds the decision of bar i-1
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[1])  # bar 0 had NaN input
        assert out.iloc[2] == 0.0  # bar 1: 1.0 != 1.0 is False
        assert out.iloc[3] == 1.0  # bar 2: 2.0 != 1.0

    def test_every_operator_gives_nan_not_one_for_nan(self):
        df = _frame(C=[1.0] * 3, x=[np.nan, np.nan, np.nan])
        for op in ("==", "!=", ">", "<", ">=", "<=", "crosses_above", "crosses_below"):
            out = _plugin([_rule(op, 1.0)]).compute(df)["sig"]
            assert out.isna().all(), op

    def test_crossing_is_nan_when_previous_value_is_nan(self):
        df = _frame(C=[1.0] * 4, x=[np.nan, 2.0, 0.0, 2.0])
        out = _plugin([_rule("crosses_above", 1.0)]).compute(df)["sig"]
        # rows are shifted by one bar: row 1 holds bar 0 (NaN input), row 2 holds
        # bar 1 (previous value NaN), row 3 holds bar 2 (2.0 -> 0.0, no cross up)
        assert pd.isna(out.iloc[1]) and pd.isna(out.iloc[2])
        assert out.iloc[3] == 0.0

    def test_and_or_use_three_valued_logic(self):
        df = _frame(
            C=[1.0] * 4, a=[1.0, 1.0, 0.0, np.nan], b=[np.nan, 1.0, np.nan, np.nan]
        )

        def rule(logic):
            return {
                "output": "sig",
                "logic": logic,
                "conditions": [
                    {"column": "a", "op": ">", "value": 0.5},
                    {"column": "b", "op": ">", "value": 0.5},
                ],
            }

        and_out = _plugin([rule("AND")]).compute(df)["sig"].shift(-1)
        or_out = _plugin([rule("OR")]).compute(df)["sig"].shift(-1)
        # AND: (T,NaN)->NaN, (T,T)->1, (F,NaN)->0 (decided), (NaN,NaN)->NaN
        assert pd.isna(and_out.iloc[0]) and and_out.iloc[1] == 1.0
        assert and_out.iloc[2] == 0.0 and pd.isna(and_out.iloc[3])
        # OR: (T,NaN)->1 (decided), (T,T)->1, (F,NaN)->NaN, (NaN,NaN)->NaN
        assert or_out.iloc[0] == 1.0 and or_out.iloc[1] == 1.0
        assert pd.isna(or_out.iloc[2]) and pd.isna(or_out.iloc[3])

    def test_complete_input_is_unchanged_binary_signal(self):
        df = _frame(C=[1.0] * 5, x=[0.0, 2.0, 0.0, 3.0, 4.0])
        out = _plugin([_rule(">", 1.0)]).compute(df)["sig"]
        assert out.iloc[1:].tolist() == [0.0, 1.0, 0.0, 1.0]
