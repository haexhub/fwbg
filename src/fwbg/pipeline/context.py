"""Pipeline context for passing data between phases."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd


@dataclass
class PipelineContext:
    """
    Context object passed through all pipeline phases.

    Carries the DataFrame and metadata that plugins can read/write.
    Each plugin receives this context, processes it, and returns
    an updated context (or the same one with modified df/metadata).

    Attributes:
        df: The main DataFrame being processed
        symbol: Asset symbol (e.g., "EURUSD")
        asset_class: Asset class (e.g., "FOREX", "CRYPTO")
        metadata: Arbitrary key-value store for inter-plugin communication
        fold_info: Optional fold information for walk-forward validation
    """
    df: pd.DataFrame
    symbol: str
    asset_class: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    fold_info: Optional[Dict[str, Any]] = None

    def clone(self) -> "PipelineContext":
        """Create a shallow copy with a new DataFrame copy."""
        return PipelineContext(
            df=self.df.copy(),
            symbol=self.symbol,
            asset_class=self.asset_class,
            metadata=self.metadata.copy(),
            fold_info=self.fold_info.copy() if self.fold_info else None,
        )
