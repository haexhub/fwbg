"""Configuration parsers for pipeline plugins and strategy configs."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PluginConfig:
    """
    Configuration for a single plugin instance.

    Attributes:
        name: Plugin name (must match a registered plugin)
        params: Plugin-specific parameters
        stateful: Override default stateful behavior (None = use plugin default)
        cacheable: Override default cacheable behavior (None = use plugin default)
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    stateful: Optional[bool] = None
    cacheable: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginConfig":
        """
        Parse a PluginConfig from a dictionary.

        Args:
            data: Dictionary with plugin configuration

        Returns:
            PluginConfig instance

        Raises:
            ValueError: If 'name' key is missing
        """
        if "name" not in data:
            raise ValueError("Plugin config must have 'name' key")

        return cls(
            name=data["name"],
            params=data.get("params", {}),
            stateful=data.get("stateful"),
            cacheable=data.get("cacheable"),
        )


@dataclass
class PipelineConfig:
    """
    Configuration for an entire pipeline with all phases.

    Attributes:
        data_loading: Plugins for data loading phase
        preprocessing: Plugins for preprocessing phase
        indicators: Plugins for indicator calculation phase
        feature_selection: Plugins for feature selection phase
        labeling: Plugins for labeling phase
        model: Plugins for model phase
        validation: Plugins for validation phase
    """

    data_loading: List[PluginConfig] = field(default_factory=list)
    preprocessing: List[PluginConfig] = field(default_factory=list)
    indicators: List[PluginConfig] = field(default_factory=list)
    feature_selection: List[PluginConfig] = field(default_factory=list)
    labeling: List[PluginConfig] = field(default_factory=list)
    model: List[PluginConfig] = field(default_factory=list)
    validation: List[PluginConfig] = field(default_factory=list)

    def get_phase(self, phase_name: str) -> List[PluginConfig]:
        """
        Get plugins for a specific phase.

        Args:
            phase_name: Name of the phase (e.g., 'indicators', 'model')

        Returns:
            List of PluginConfig instances for the phase,
            or empty list if phase doesn't exist
        """
        return getattr(self, phase_name, [])

    def all_plugins(self) -> List[PluginConfig]:
        """
        Get all plugins across all phases in execution order.

        Returns:
            List of all PluginConfig instances
        """
        return (
            self.data_loading
            + self.preprocessing
            + self.indicators
            + self.feature_selection
            + self.labeling
            + self.model
            + self.validation
        )


def parse_pipeline_config(data: dict) -> PipelineConfig:
    """
    Parse a strategy dictionary into a PipelineConfig.

    Args:
        data: Strategy dictionary with 'pipeline' section

    Returns:
        PipelineConfig instance

    Raises:
        ValueError: If any plugin config is invalid
    """
    pipeline_data = data.get("pipeline", {})

    def parse_phase(phase_name: str) -> List[PluginConfig]:
        """Parse a list of plugin configs for a phase."""
        phase_data = pipeline_data.get(phase_name, [])
        return [PluginConfig.from_dict(plugin_dict) for plugin_dict in phase_data]

    return PipelineConfig(
        data_loading=parse_phase("data_loading"),
        preprocessing=parse_phase("preprocessing"),
        indicators=parse_phase("indicators"),
        feature_selection=parse_phase("feature_selection"),
        labeling=parse_phase("labeling"),
        model=parse_phase("model"),
        validation=parse_phase("validation"),
    )
