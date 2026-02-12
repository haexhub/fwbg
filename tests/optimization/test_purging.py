"""
Tests für Purged CV und Sample Weights (López de Prado, AFML Ch. 4+7).

Testet:
- ValidationConfig embargo_bars und sample_weights Felder
- Embargo-Gap in Fold-Erstellung (outer + inner folds)
- Concurrent Label Counting und Sample Uniqueness Weights
"""
import pytest
import numpy as np
import pandas as pd

from fwbg.core.config import ValidationConfig


class TestValidationConfigEmbargo:
    def test_embargo_bars_from_dict(self):
        config = ValidationConfig.from_dict({"embargo_bars": 100})
        assert config.embargo_bars == 100

    def test_embargo_bars_default(self):
        config = ValidationConfig.from_dict({})
        assert config.embargo_bars == 0

    def test_sample_weights_from_dict(self):
        config = ValidationConfig.from_dict({"sample_weights": True})
        assert config.sample_weights is True

    def test_sample_weights_default(self):
        config = ValidationConfig.from_dict({})
        assert config.sample_weights is False
