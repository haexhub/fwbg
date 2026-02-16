"""Base plugin class and phase enum for the pipeline system."""
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from fwbg_sdk.contexts import PipelineContext


class PluginPhase(Enum):
    """Pipeline phases in execution order."""

    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    INDICATORS = "indicators"
    FEATURE_SELECTION = "feature_selection"
    EXIT_STRATEGIES = "exit_strategies"
    RISK_MANAGEMENT = "risk_management"
    LABELING = "labeling"
    MODEL = "model"
    VALIDATION = "validation"


class BasePlugin(ABC):
    """
    Abstract base class for all pipeline plugins.

    Subclasses must define:
        - name: str - unique identifier for the plugin
        - phase: PluginPhase - which pipeline phase this plugin belongs to

    Optional class attributes:
        - version: str - semantic version string (default: "0.1.0")
        - stateful: bool - whether plugin maintains state across calls (default: False)
        - cacheable: bool - whether results can be cached (default: True)
    """

    # Required class attributes (must be defined by subclasses)
    name: str
    phase: PluginPhase
    version: str = "0.1.0"

    # Optional class attributes with defaults
    stateful: bool = False
    cacheable: bool = True
    depends_on: List[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that required class attributes are defined."""
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract subclasses (have ABC in their bases or abstractmethods)
        if ABC in cls.__bases__:
            return

        # Skip if this is an intermediate base class (has abstract methods)
        if getattr(cls, "__abstractmethods__", None):
            return

        # Check required attributes - can be inherited from parent or defined in class
        if not hasattr(cls, "name") or not isinstance(getattr(cls, "name", None), str):
            raise TypeError(
                f"Plugin class {cls.__name__} must define 'name' attribute"
            )

        if not hasattr(cls, "phase") or not isinstance(getattr(cls, "phase", None), PluginPhase):
            raise TypeError(
                f"Plugin class {cls.__name__} must define 'phase' attribute"
            )

    def __init__(self) -> None:
        """Initialize plugin instance state."""
        self._fitted: bool = False

    def execute(
        self, ctx: "PipelineContext", **params: Any
    ) -> "PipelineContext":
        """Execute the plugin on the given context.

        Subclasses like BaseIndicator override this. Plugin types that don't
        use the PipelineRunner (exit strategies, risk managers) can leave the default.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement execute()"
        )

    def validate(self) -> bool:
        """Validate that the plugin is properly configured."""
        return True

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

    @classmethod
    def get_test_module_path(cls) -> Optional[Path]:
        """
        Get the path to this plugin's test module.

        Returns:
            Path to tests.py if it exists, None otherwise
        """
        import inspect
        module = inspect.getmodule(cls)
        if module is None or module.__file__ is None:
            return None

        plugin_dir = Path(module.__file__).parent
        test_file = plugin_dir / "tests.py"
        return test_file if test_file.exists() else None

    @classmethod
    def run_tests(cls, verbose: bool = False) -> Tuple[int, int, List[str]]:
        """
        Run the plugin's test suite.

        Each plugin should have a tests.py file in its directory with
        pytest-compatible test functions.

        Args:
            verbose: If True, print verbose test output

        Returns:
            Tuple of (passed, failed, error_messages)
        """
        import importlib.util
        import sys

        test_path = cls.get_test_module_path()
        if test_path is None:
            return (0, 0, [f"No tests.py found for plugin {cls.name}"])

        # Load the test module
        module_name = f"fwbg.plugins.tests.{cls.name}"
        spec = importlib.util.spec_from_file_location(module_name, test_path)
        if spec is None or spec.loader is None:
            return (0, 0, [f"Could not load test module from {test_path}"])

        test_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = test_module

        try:
            spec.loader.exec_module(test_module)
        except Exception as e:
            return (0, 0, [f"Error loading test module: {e}"])

        # Run pytest on the test module
        try:
            import pytest
            args = [str(test_path)]
            if verbose:
                args.append("-v")
            result = pytest.main(args)

            if result == 0:
                return (1, 0, [])
            else:
                return (0, 1, [f"Tests failed with exit code {result}"])
        except ImportError:
            # Fallback: run tests manually without pytest
            passed = 0
            failed = 0
            errors: List[str] = []

            for name in dir(test_module):
                if name.startswith("test_"):
                    test_func = getattr(test_module, name)
                    if callable(test_func):
                        try:
                            test_func()
                            passed += 1
                            if verbose:
                                print(f"  ✓ {name}")
                        except AssertionError as e:
                            failed += 1
                            errors.append(f"{name}: {e}")
                            if verbose:
                                print(f"  ✗ {name}: {e}")
                        except Exception as e:
                            failed += 1
                            errors.append(f"{name}: {e}")
                            if verbose:
                                print(f"  ✗ {name}: {e}")

            return (passed, failed, errors)

    @classmethod
    def has_tests(cls) -> bool:
        """Check if this plugin has a test suite."""
        return cls.get_test_module_path() is not None
