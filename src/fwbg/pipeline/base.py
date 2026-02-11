"""Base plugin class and phase enum for the pipeline system."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from fwbg.pipeline.context import PipelineContext


class PluginPhase(Enum):
    """Pipeline phases in execution order."""

    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    INDICATORS = "indicators"
    FEATURE_SELECTION = "feature_selection"
    LABELING = "labeling"
    MODEL = "model"
    VALIDATION = "validation"


class BasePlugin(ABC):
    """
    Abstract base class for all pipeline plugins.

    Subclasses must define:
        - name: str - unique identifier for the plugin
        - version: str - semantic version string
        - phase: PluginPhase - which pipeline phase this plugin belongs to

    Optional class attributes:
        - stateful: bool - whether plugin maintains state across calls (default: False)
        - cacheable: bool - whether results can be cached (default: True)
    """

    # Required class attributes (must be defined by subclasses)
    name: str
    version: str
    phase: PluginPhase

    # Optional class attributes with defaults
    stateful: bool = False
    cacheable: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that required class attributes are defined."""
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract subclasses
        if ABC in cls.__bases__:
            return

        # Check required attributes - must be actual values, not just annotations
        # Use vars(cls) to check if attribute is defined in this class (not inherited annotation)
        cls_attrs = vars(cls)

        # name must be defined as actual value
        if "name" not in cls_attrs or not isinstance(cls_attrs.get("name"), str):
            raise TypeError(
                f"Plugin class {cls.__name__} must define 'name' attribute"
            )

        # version must be defined as actual value
        if "version" not in cls_attrs or not isinstance(cls_attrs.get("version"), str):
            raise TypeError(
                f"Plugin class {cls.__name__} must define 'version' attribute"
            )

        # phase must be defined as actual value
        if "phase" not in cls_attrs or not isinstance(cls_attrs.get("phase"), PluginPhase):
            raise TypeError(
                f"Plugin class {cls.__name__} must define 'phase' attribute"
            )

    def __init__(self) -> None:
        """Initialize plugin instance state."""
        self._fitted: bool = False

    @abstractmethod
    def execute(
        self, ctx: "PipelineContext", **params: Any
    ) -> "PipelineContext":
        """
        Execute the plugin on the given context.

        Args:
            ctx: Pipeline context with DataFrame and metadata
            **params: Plugin-specific parameters

        Returns:
            Updated pipeline context
        """
        ...

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate that the plugin is properly configured.

        Returns:
            True if valid, False otherwise
        """
        ...

    def fit(self, ctx: "PipelineContext", **params: Any) -> None:
        """
        Fit the plugin to the given context (for stateful plugins).

        Args:
            ctx: Pipeline context with training data
            **params: Plugin-specific parameters
        """
        self._fitted = True

    def reset(self) -> None:
        """Reset the plugin to its initial unfitted state."""
        self._fitted = False

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for this plugin.

        Returns:
            Dictionary of parameter names to default values
        """
        return {}

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns created by this plugin.

        Returns:
            List of column names
        """
        return []

    def report_progress(
        self,
        current: int,
        total: int,
        message: str = "",
        callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        """
        Report progress during plugin execution.

        Args:
            current: Current progress value
            total: Total progress value
            message: Optional progress message
            callback: Optional callback function to invoke with progress info
        """
        if callback is not None:
            callback(current=current, total=total, message=message)
