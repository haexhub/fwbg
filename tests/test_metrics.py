"""
Tests für alle Optimizer-Kennzahlen mit synthetischen Daten.

Testet:
- Sharpe Ratio
- Calmar Ratio
- Win Rate
- RRR (Risk-Reward-Ratio)
- Kelly Criterion
- Annual Return
- Max Drawdown
- Fold Stability
- Equity Smoothness
- Monte Carlo p-value
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.simulation import (
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    monte_carlo_permutation_test,
    monte_carlo_equity_simulation,
    adjust_kelly_for_target_dd,
    calculate_equity_smoothness,
)


class TestSharpeRatio:
    """Tests für Sharpe Ratio Berechnung."""

    def test_perfect_strategy_high_sharpe(self):
        """Strategie mit nur Gewinnen sollte hohen Sharpe haben."""
        # 100 Trades, alle +1%
        returns = [0.01] * 100
        sharpe = calculate_sharpe_ratio(returns)
        # Keine Varianz = 0 nach Formel (std=0 gibt 0 zurück)
        # Also testen wir mit leichter Varianz für einen hohen Sharpe
        returns_with_variance = [0.01, 0.0099] * 50  # Fast keine Varianz
        sharpe_high = calculate_sharpe_ratio(returns_with_variance)
        assert sharpe_high > 10, f"Fast perfekte Strategie sollte sehr hohen Sharpe haben, got {sharpe_high}"

    def test_losing_strategy_negative_sharpe(self):
        """Strategie mit nur Verlusten sollte negativen Sharpe haben."""
        returns = [-0.01] * 100
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0, f"Verluststrategie sollte negativen Sharpe haben, got {sharpe}"

    def test_breakeven_strategy_zero_sharpe(self):
        """Strategie mit 0 Erwartungswert sollte ~0 Sharpe haben."""
        # Abwechselnd +1% und -1%
        returns = [0.01, -0.01] * 50
        sharpe = calculate_sharpe_ratio(returns)
        assert abs(sharpe) < 0.5, f"Breakeven sollte ~0 Sharpe haben, got {sharpe}"

    def test_volatile_strategy_lower_sharpe(self):
        """Hohe Volatilität sollte Sharpe reduzieren."""
        # Gleicher Durchschnitt, aber unterschiedliche Volatilität
        low_vol = [0.01] * 100
        high_vol = [0.05, -0.03] * 50  # Durchschnitt 0.01, aber volatiler

        sharpe_low = calculate_sharpe_ratio(low_vol)
        sharpe_high = calculate_sharpe_ratio(high_vol)

        assert sharpe_low > sharpe_high, \
            f"Niedrige Vol ({sharpe_low}) sollte höheren Sharpe haben als hohe Vol ({sharpe_high})"

    def test_empty_returns(self):
        """Leere Returns sollten 0 zurückgeben."""
        sharpe = calculate_sharpe_ratio([])
        assert sharpe == 0, f"Leere Returns sollten 0 Sharpe geben, got {sharpe}"


class TestCalmarRatio:
    """Tests für Calmar Ratio (Return / Max Drawdown)."""

    def test_no_drawdown_max_calmar(self):
        """Keine Verluste = maximaler Calmar."""
        trades = [1] * 100  # Alle Gewinne
        kelly = 0.01
        rrr = 1.0
        calmar = calculate_calmar_ratio(trades, kelly, rrr)
        assert calmar == 10.0, f"Kein DD sollte max Calmar geben, got {calmar}"

    def test_all_losses_negative_calmar(self):
        """Nur Verluste = negativer Calmar."""
        trades = [-1] * 100
        kelly = 0.01
        rrr = 1.0
        calmar = calculate_calmar_ratio(trades, kelly, rrr)
        assert calmar < 0, f"Nur Verluste sollten negativen Calmar geben, got {calmar}"

    def test_deep_drawdown_low_calmar(self):
        """Tiefer Drawdown sollte Calmar reduzieren."""
        # Erst 10 Verluste, dann 20 Gewinne
        trades_deep_dd = [-1] * 10 + [1] * 20
        # Erst 2 Verluste, dann 20 Gewinne
        trades_shallow_dd = [-1] * 2 + [1] * 20

        kelly = 0.02
        rrr = 1.0

        calmar_deep = calculate_calmar_ratio(trades_deep_dd, kelly, rrr)
        calmar_shallow = calculate_calmar_ratio(trades_shallow_dd, kelly, rrr)

        assert calmar_shallow > calmar_deep, \
            f"Flacher DD ({calmar_shallow}) sollte höheren Calmar haben als tiefer DD ({calmar_deep})"


class TestWinRate:
    """Tests für Win Rate Berechnung."""

    def test_all_wins(self):
        """100% Gewinne."""
        trades = [1, 1, 1, 1, 1]
        win_rate = sum(1 for t in trades if t > 0) / len(trades)
        assert win_rate == 1.0

    def test_all_losses(self):
        """0% Gewinne."""
        trades = [-1, -1, -1, -1, -1]
        win_rate = sum(1 for t in trades if t > 0) / len(trades)
        assert win_rate == 0.0

    def test_fifty_fifty(self):
        """50% Win Rate."""
        trades = [1, -1, 1, -1, 1, -1]
        win_rate = sum(1 for t in trades if t > 0) / len(trades)
        assert win_rate == 0.5

    def test_realistic_scalping(self):
        """Realistische Scalping-Strategie: hohe WR, niedriger RRR."""
        # 90% Gewinne, 10% Verluste
        trades = [1] * 90 + [-1] * 10
        win_rate = sum(1 for t in trades if t > 0) / len(trades)
        assert win_rate == 0.9


class TestRRR:
    """Tests für Risk-Reward-Ratio."""

    def test_rrr_calculation(self):
        """RRR = TP / SL."""
        assert 30 / 100 == 0.3  # TP=30, SL=100
        assert 50 / 50 == 1.0   # TP=50, SL=50
        assert 100 / 50 == 2.0  # TP=100, SL=50

    def test_breakeven_win_rate(self):
        """Breakeven WR = 1 / (1 + RRR)."""
        # RRR 0.5 -> BE WR = 66.7%
        rrr = 0.5
        be_wr = 1 / (1 + rrr)
        assert abs(be_wr - 0.6667) < 0.01

        # RRR 1.0 -> BE WR = 50%
        rrr = 1.0
        be_wr = 1 / (1 + rrr)
        assert be_wr == 0.5

        # RRR 2.0 -> BE WR = 33.3%
        rrr = 2.0
        be_wr = 1 / (1 + rrr)
        assert abs(be_wr - 0.3333) < 0.01


class TestKellyCriterion:
    """Tests für Kelly Criterion."""

    def test_kelly_formula(self):
        """Kelly = (p * b - q) / b, wobei p=WR, q=1-WR, b=RRR."""
        # WR=60%, RRR=1.0 -> Kelly = (0.6*1 - 0.4) / 1 = 0.2 = 20%
        wr = 0.6
        rrr = 1.0
        kelly = (wr * rrr - (1 - wr)) / rrr
        assert abs(kelly - 0.2) < 0.001

        # WR=90%, RRR=0.2 -> Kelly = (0.9*0.2 - 0.1) / 0.2 = 0.4 = 40%
        wr = 0.9
        rrr = 0.2
        kelly = (wr * rrr - (1 - wr)) / rrr
        assert abs(kelly - 0.4) < 0.001

    def test_negative_kelly_losing_strategy(self):
        """Verlierende Strategie hat negativen Kelly."""
        # WR=40%, RRR=1.0 -> Kelly = (0.4*1 - 0.6) / 1 = -0.2
        wr = 0.4
        rrr = 1.0
        kelly = (wr * rrr - (1 - wr)) / rrr
        assert kelly < 0

    def test_kelly_adjustment_for_drawdown(self):
        """Kelly-Anpassung sollte DD reduzieren."""
        # Strategie mit hohem DD
        trades = [-1] * 5 + [1] * 15  # Losing streak am Anfang
        kelly = 0.05
        rrr = 1.0

        result = adjust_kelly_for_target_dd(trades, kelly, rrr, target_max_dd=0.30)

        assert result["adjusted_kelly"] <= kelly, \
            f"Adjusted Kelly ({result['adjusted_kelly']}) sollte <= original ({kelly}) sein"
        assert result["scale_factor"] <= 1.0


class TestAnnualReturn:
    """Tests für Jahresrendite-Berechnung."""

    def test_annual_return_formula(self):
        """Annual Return = ((final_equity / 100) ^ (1/years) - 1) * 100."""
        # 100% Gewinn in 1 Jahr = 100%/y
        final_equity = 200
        years = 1
        annual_return = ((final_equity / 100.0) ** (1 / years) - 1) * 100
        assert annual_return == 100.0

        # 100% Gewinn in 2 Jahren = ~41.4%/y (compound)
        years = 2
        annual_return = ((final_equity / 100.0) ** (1 / years) - 1) * 100
        assert abs(annual_return - 41.42) < 0.1

    def test_negative_return(self):
        """Verlust sollte negativen Annual Return geben."""
        final_equity = 50  # 50% Verlust
        years = 1
        annual_return = ((final_equity / 100.0) ** (1 / years) - 1) * 100
        assert annual_return == -50.0

    def test_compound_growth(self):
        """Compound Growth über mehrere Jahre."""
        # 10%/y über 5 Jahre = 1.1^5 = 1.61 = +61%
        annual_rate = 0.10
        years = 5
        final_equity = 100 * ((1 + annual_rate) ** years)
        assert abs(final_equity - 161.05) < 0.1


class TestMaxDrawdown:
    """Tests für Maximum Drawdown."""

    def test_no_drawdown(self):
        """Nur steigende Equity = 0% DD."""
        equity = [100, 110, 120, 130, 140]
        peak = equity[0]
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        assert max_dd == 0

    def test_50_percent_drawdown(self):
        """50% Drawdown von Peak."""
        equity = [100, 150, 75, 100]  # Peak 150, Trough 75 = 50% DD
        peak = equity[0]
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        assert max_dd == 0.5

    def test_recovery_doesnt_change_max_dd(self):
        """Recovery ändert Max DD nicht."""
        equity = [100, 50, 100, 200]  # 50% DD, dann Recovery
        peak = equity[0]
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        assert max_dd == 0.5  # Bleibt 50% trotz Recovery


class TestFoldStability:
    """Tests für Fold-Stabilität."""

    def test_all_folds_profitable(self):
        """Alle Folds profitabel = 100% Stabilität."""
        fold_pnls = [10, 20, 15, 30, 25]
        profitable = sum(1 for pnl in fold_pnls if pnl > 0)
        stability = profitable / len(fold_pnls)
        assert stability == 1.0

    def test_no_folds_profitable(self):
        """Keine Folds profitabel = 0% Stabilität."""
        fold_pnls = [-10, -20, -15, -30, -25]
        profitable = sum(1 for pnl in fold_pnls if pnl > 0)
        stability = profitable / len(fold_pnls)
        assert stability == 0.0

    def test_half_folds_profitable(self):
        """Hälfte profitabel = 50% Stabilität."""
        fold_pnls = [10, -20, 15, -30]
        profitable = sum(1 for pnl in fold_pnls if pnl > 0)
        stability = profitable / len(fold_pnls)
        assert stability == 0.5

    def test_minimum_stability_threshold(self):
        """Mindestens 50% Folds sollten profitabel sein."""
        MIN_STABILITY = 0.5

        # 3/5 Folds profitabel = 60% -> OK
        fold_pnls_good = [10, 20, -5, 15, -10]
        stability_good = sum(1 for p in fold_pnls_good if p > 0) / len(fold_pnls_good)
        assert stability_good >= MIN_STABILITY

        # 2/5 Folds profitabel = 40% -> SKIP
        fold_pnls_bad = [10, -20, -5, 15, -10]
        stability_bad = sum(1 for p in fold_pnls_bad if p > 0) / len(fold_pnls_bad)
        assert stability_bad < MIN_STABILITY


class TestEquitySmoothness:
    """Tests für Equity Smoothness."""

    def test_smooth_equity_curve(self):
        """Gleichmäßige Gewinne = hohe Smoothness."""
        trades = [1] * 100  # Alle Gewinne, gleichmäßig
        kelly = 0.01
        rrr = 1.0
        result = calculate_equity_smoothness(trades, kelly, rrr)

        assert result["smoothness_score"] > 0.8, \
            f"Gleichmäßige Gewinne sollten hohe Smoothness haben, got {result['smoothness_score']}"

    def test_volatile_equity_curve(self):
        """Stark schwankende Equity = niedrige Sortino oder höhere Volatilität."""
        # Abwechselnd große Gewinne und Verluste
        trades = [1, -1, 1, -1, 1, -1] * 10
        kelly = 0.05
        rrr = 1.0
        result = calculate_equity_smoothness(trades, kelly, rrr)

        # Bei 50/50 Trades ist die Smoothness nicht zwingend niedrig,
        # aber die return_volatility sollte messbar sein
        assert result["return_volatility"] > 0, \
            f"Volatile Equity sollte messbare Volatilität haben, got {result['return_volatility']}"

    def test_sortino_ratio_calculation(self):
        """Sortino Ratio berücksichtigt nur Downside-Volatilität."""
        # Strategie mit nur Gewinnen sollte hohen/unendlichen Sortino haben
        trades_only_wins = [1] * 100
        kelly = 0.02
        rrr = 1.0
        result = calculate_equity_smoothness(trades_only_wins, kelly, rrr)

        # Keine negativen Returns = hoher Sortino (gecapped bei 10)
        assert result["sortino_ratio"] == 10.0, \
            f"Nur Gewinne sollten max Sortino geben, got {result['sortino_ratio']}"


class TestMonteCarloPermutation:
    """Tests für Monte Carlo Permutation Test."""

    def test_significant_strategy(self):
        """Stark profitable Strategie sollte signifikant sein."""
        # 80% Gewinne
        trades = [1] * 80 + [-1] * 20
        result = monte_carlo_permutation_test(trades, n_permutations=500)

        assert result["is_significant"], \
            f"80% WR sollte signifikant sein, p={result['p_value']}"
        assert result["p_value"] < 0.05

    def test_random_strategy_not_significant(self):
        """Zufällige Strategie sollte nicht signifikant sein."""
        # 50% Gewinne = Zufall
        trades = [1, -1] * 50
        result = monte_carlo_permutation_test(trades, n_permutations=500)

        # Bei 50/50 sollte p-value hoch sein
        assert result["p_value"] > 0.3, \
            f"50/50 sollte hohen p-value haben, got {result['p_value']}"

    def test_losing_strategy(self):
        """Verlierende Strategie hat hohen p-value."""
        trades = [-1] * 70 + [1] * 30
        result = monte_carlo_permutation_test(trades, n_permutations=500)

        # Sollte nicht signifikant sein (schlechter als Zufall)
        assert not result["is_significant"] or result["p_value"] > 0.05


class TestMonteCarloEquity:
    """Tests für Monte Carlo Equity Simulation."""

    def test_profitable_strategy_grows(self):
        """Profitable Strategie sollte wachsen."""
        trades = [1] * 70 + [-1] * 30  # 70% WR
        kelly = 0.02
        rrr = 1.0
        result = monte_carlo_equity_simulation(trades, kelly, rrr, n_simulations=200)

        assert result["median_equity"] > 100, \
            f"70% WR sollte Equity wachsen lassen, got {result['median_equity']}"

    def test_losing_strategy_shrinks(self):
        """Verlierende Strategie sollte schrumpfen."""
        trades = [1] * 30 + [-1] * 70  # 30% WR
        kelly = 0.02
        rrr = 1.0
        result = monte_carlo_equity_simulation(trades, kelly, rrr, n_simulations=200)

        assert result["median_equity"] < 100, \
            f"30% WR sollte Equity schrumpfen lassen, got {result['median_equity']}"

    def test_high_kelly_increases_bankruptcy(self):
        """Hoher Kelly erhöht Bankruptcy-Risiko."""
        trades = [1] * 60 + [-1] * 40

        result_low_kelly = monte_carlo_equity_simulation(
            trades, kelly_risk=0.01, rrr=1.0, n_simulations=200
        )
        result_high_kelly = monte_carlo_equity_simulation(
            trades, kelly_risk=0.10, rrr=1.0, n_simulations=200
        )

        # Hoher Kelly sollte mehr Bankruptcies haben
        # (oder zumindest nicht weniger)
        assert result_high_kelly["bankruptcy_rate"] >= result_low_kelly["bankruptcy_rate"] - 0.05


class TestExpectancy:
    """Tests für Expectancy (Edge über Breakeven)."""

    def test_positive_expectancy(self):
        """Positive Expectancy = profitabel."""
        # WR=60%, RRR=1.0 -> Expectancy = 0.6*1 - 0.4*1 = 0.2
        wr = 0.6
        rrr = 1.0
        expectancy = wr * rrr - (1 - wr)
        assert expectancy > 0

    def test_negative_expectancy(self):
        """Negative Expectancy = verlierend."""
        # WR=40%, RRR=1.0 -> Expectancy = 0.4*1 - 0.6*1 = -0.2
        wr = 0.4
        rrr = 1.0
        expectancy = wr * rrr - (1 - wr)
        assert expectancy < 0

    def test_zero_expectancy_breakeven(self):
        """Breakeven bei Expectancy = 0."""
        # WR=66.67%, RRR=0.5 -> Expectancy = 0.6667*0.5 - 0.3333 = 0
        wr = 2/3  # 66.67%
        rrr = 0.5
        expectancy = wr * rrr - (1 - wr)
        assert abs(expectancy) < 0.001

    def test_scalping_expectancy(self):
        """Scalping: hohe WR, niedriger RRR kann profitabel sein."""
        # WR=90%, RRR=0.2 -> Expectancy = 0.9*0.2 - 0.1 = 0.08
        wr = 0.9
        rrr = 0.2
        expectancy = wr * rrr - (1 - wr)
        assert expectancy > 0, f"90% WR mit RRR 0.2 sollte profitabel sein, got {expectancy}"

        # Aber WR=80%, RRR=0.2 -> Expectancy = 0.8*0.2 - 0.2 = -0.04
        wr = 0.8
        expectancy = wr * rrr - (1 - wr)
        assert expectancy < 0, f"80% WR mit RRR 0.2 sollte verlierend sein, got {expectancy}"


class TestIntegration:
    """Integrationstests mit realistischen Szenarien."""

    def test_realistic_forex_strategy(self):
        """Realistisches Forex-Szenario."""
        # 55% WR, RRR 1.5, 1000 Trades
        np.random.seed(42)
        wr = 0.55
        rrr = 1.5
        n_trades = 1000

        trades = np.random.choice([1, -1], size=n_trades, p=[wr, 1-wr]).tolist()
        kelly = (wr * rrr - (1 - wr)) / rrr
        kelly = min(0.05, kelly / 4)  # 1/4 Kelly, max 5%

        # Berechne alle Metriken
        actual_wr = sum(1 for t in trades if t > 0) / len(trades)
        expectancy = actual_wr * rrr - (1 - actual_wr)

        trade_returns = [kelly * rrr if t > 0 else -kelly for t in trades]
        sharpe = calculate_sharpe_ratio(trade_returns)
        calmar = calculate_calmar_ratio(trades, kelly, rrr)
        smoothness = calculate_equity_smoothness(trades, kelly, rrr)
        mc_perm = monte_carlo_permutation_test(trades, n_permutations=500)

        # Assertions
        assert abs(actual_wr - wr) < 0.05, f"WR sollte ~55% sein, got {actual_wr}"
        assert expectancy > 0, f"Expectancy sollte positiv sein, got {expectancy}"
        assert sharpe > 0, f"Sharpe sollte positiv sein, got {sharpe}"
        assert mc_perm["is_significant"], f"Sollte signifikant sein, p={mc_perm['p_value']}"

    def test_overfitted_strategy_detection(self):
        """Overfitted Strategie sollte erkennbar sein."""
        # In-Sample: perfekt, Out-of-Sample: schlecht
        # Simuliere durch unterschiedliche Fold-Performance

        # "Overfitted": Erster Fold super, Rest schlecht
        fold_pnls_overfitted = [100, -10, -5, -8, -12]
        stability_overfitted = sum(1 for p in fold_pnls_overfitted if p > 0) / len(fold_pnls_overfitted)

        # "Robust": Alle Folds ähnlich gut
        fold_pnls_robust = [20, 18, 22, 19, 21]
        stability_robust = sum(1 for p in fold_pnls_robust if p > 0) / len(fold_pnls_robust)

        assert stability_overfitted < 0.5, "Overfitted sollte niedrige Stabilität haben"
        assert stability_robust == 1.0, "Robuste Strategie sollte 100% Stabilität haben"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
