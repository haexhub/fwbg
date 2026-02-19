# Correlation Filter Feature Selection

Greedy redundancy filter that removes highly correlated features, keeping the most important ones based on input order.

## Concept

Feature sets produced by importance-based selectors (such as Boruta or Plateau) often contain groups of highly correlated features -- for example, `rsi_14` and `rsi_12`, or `macro_vix_chg_24h` and `macro_vix_chg_12h`. While each may be individually important, including multiple redundant features wastes model capacity, increases overfitting risk, and can destabilize tree-based models by splitting importance across near-identical signals.

The correlation filter addresses this by computing the absolute pairwise correlation matrix and greedily keeping only features whose correlation with all previously kept features falls below a configurable threshold. Because features arrive pre-sorted by importance from an upstream selector, the filter always retains the most important feature in each correlated cluster and drops the less important duplicates.

This plugin is designed to be used as a second-stage selector in a feature selection pipeline, typically after Boruta, Stability, or Plateau selection. The O(n^2) greedy algorithm is efficient enough for the typical feature counts encountered after initial selection.

## Selection Algorithm

1. Compute the absolute pairwise Pearson correlation matrix of all input features
2. Iterate through features in their input order (most important first, as sorted by the upstream selector)
3. For each feature, check its absolute correlation with every already-kept feature
4. If any correlation >= `max_correlation`, drop the feature and record which kept feature caused the drop
5. Otherwise, keep the feature
6. Stop early if `max_features` limit is reached

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_correlation` | `float` | `0.7` | Maximum absolute pairwise correlation allowed between kept features (min: 0.0, max: 1.0, step: 0.05) |
| `max_features` | `int` | `None` | Optional hard cap on the number of output features |

## Usage Notes

- Input features should be pre-sorted by importance (most important first). The filter relies on this ordering to keep the best feature in each correlated group.
- The `y` (target) parameter is required by the selector interface but is not used by this plugin.
- If the input contains 0 or 1 features, the filter returns them unchanged with `n_dropped: 0`.
- Metadata includes the full list of dropped features and their drop reasons (which kept feature caused the drop and the correlation value), which is useful for debugging feature selection pipelines.
- A threshold of `0.7` is a common default in financial ML. Lower values (e.g., `0.5`) produce more aggressive filtering; higher values (e.g., `0.85`) preserve more features at the cost of greater redundancy.
