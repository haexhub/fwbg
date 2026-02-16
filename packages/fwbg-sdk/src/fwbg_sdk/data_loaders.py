"""Base class for data loading plugins.

DataLoaders compute features from already-loaded raw data.
They do NO I/O — raw data is already in the DataFrame
(loaded by the orchestrator via DataSource).
"""
from abc import ABC, abstractmethod

from fwbg_sdk.base import BasePlugin, PluginPhase


class BaseDataLoader(BasePlugin, ABC):
    """Base class for data loading plugins (pure computation, no I/O)."""

    phase = PluginPhase.DATA_LOADING
    stateful = False

    @abstractmethod
    def execute(self, ctx, **params):
        """Compute derived features from raw data in ctx.df."""
        ...

    def validate(self) -> bool:
        return True
