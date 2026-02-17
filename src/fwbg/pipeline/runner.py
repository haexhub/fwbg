"""Pipeline runner for orchestrating plugin execution in phase order."""
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from fwbg_sdk import BasePlugin, PluginPhase, PipelineContext
from fwbg.pipeline.config import PipelineConfig, PluginConfig
from fwbg.pipeline.registry import PluginRegistry


def _short_name(fq_name: str) -> str:
    """Extract short name from fully-qualified plugin name ('fwbg-premium:regime' → 'regime')."""
    return fq_name.split(":")[-1] if ":" in fq_name else fq_name


def _topological_sort(
    plugin_configs: List[PluginConfig],
    registry: PluginRegistry,
) -> List[PluginConfig]:
    """Sort plugins within a phase by depends_on using Kahn's algorithm.

    Validates that all dependencies are present and detects cycles.

    Args:
        plugin_configs: Plugin configs for a single phase
        registry: Plugin registry for looking up plugin classes

    Returns:
        Topologically sorted list of PluginConfig

    Raises:
        ValueError: If a dependency is missing or a cycle is detected
    """
    if not plugin_configs:
        return []

    # Build short_name → fq_name mapping for this phase
    short_to_fq: Dict[str, str] = {}
    for pc in plugin_configs:
        short = _short_name(pc.name)
        short_to_fq[short] = pc.name

    # Build adjacency list and in-degree count (using FQ names as node IDs)
    in_degree: Dict[str, int] = {pc.name: 0 for pc in plugin_configs}
    dependents: Dict[str, List[str]] = defaultdict(list)  # dep → [plugins that need it]

    for pc in plugin_configs:
        plugin_cls = registry.get(pc.name)
        for dep_short in plugin_cls.depends_on:
            if dep_short not in short_to_fq:
                raise ValueError(
                    f"Plugin '{_short_name(pc.name)}' depends on '{dep_short}', "
                    f"but '{dep_short}' is not in the pipeline. Either add "
                    f"'{dep_short}' to the pipeline or remove '{_short_name(pc.name)}'."
                )
            dep_fq = short_to_fq[dep_short]
            dependents[dep_fq].append(pc.name)
            in_degree[pc.name] += 1

    # Kahn's algorithm
    queue = deque(fq for fq, deg in in_degree.items() if deg == 0)
    sorted_fqs: List[str] = []

    while queue:
        node = queue.popleft()
        sorted_fqs.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_fqs) != len(plugin_configs):
        cycle_nodes = [_short_name(fq) for fq, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"Circular dependency detected among plugins: {', '.join(cycle_nodes)}"
        )

    # Return PluginConfigs in sorted order
    fq_to_config = {pc.name: pc for pc in plugin_configs}
    return [fq_to_config[fq] for fq in sorted_fqs]


class PipelineRunner:
    """
    Orchestrates plugin execution in phase order.

    Handles plugin instantiation, parameter merging, fit/transform
    workflow for stateful plugins, and execution in correct phase order.
    """

    # Phase execution order
    PHASE_ORDER: List[Tuple[str, PluginPhase]] = [
        ("data_loading", PluginPhase.DATA_LOADING),
        ("preprocessing", PluginPhase.PREPROCESSING),
        ("indicators", PluginPhase.INDICATORS),
        ("feature_selection", PluginPhase.FEATURE_SELECTION),
        ("labeling", PluginPhase.LABELING),
        ("model", PluginPhase.MODEL),
        ("validation", PluginPhase.VALIDATION),
    ]

    def __init__(
        self,
        registry: PluginRegistry,
        config: PipelineConfig,
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        """
        Initialize the pipeline runner.

        Args:
            registry: Plugin registry for looking up plugin classes
            config: Pipeline configuration with plugin configs per phase
            progress_callback: Optional callback for progress reporting
        """
        self._registry = registry
        self._config = config
        self._progress_callback = progress_callback
        self._instances: Dict[str, BasePlugin] = {}
        self._execution_order: List[Tuple[PluginConfig, BasePlugin]] = []
        self._initialized: bool = False

    @property
    def config(self) -> PipelineConfig:
        """Access to the pipeline configuration."""
        return self._config

    def _initialize(self) -> None:
        """
        Lazy initialization of plugin instances.

        Creates plugin instances in phase order, validates dependencies,
        and topologically sorts plugins within each phase.
        """
        if self._initialized:
            return

        self._instances.clear()
        self._execution_order.clear()

        # Build execution order by iterating phases in order
        for phase_name, _phase_enum in self.PHASE_ORDER:
            phase_configs = self._config.get_phase(phase_name)

            # Topological sort handles dependency validation + ordering
            sorted_configs = _topological_sort(phase_configs, self._registry)

            for plugin_config in sorted_configs:
                plugin_cls = self._registry.get(plugin_config.name)
                instance = plugin_cls()
                self._instances[plugin_config.name] = instance
                self._execution_order.append((plugin_config, instance))

        self._initialized = True

    def _merge_params(
        self,
        plugin_cls: Type[BasePlugin],
        config_params: Dict[str, Any],
        global_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge parameters with priority: global_params > config_params > default_params.

        Args:
            plugin_cls: Plugin class for getting defaults
            config_params: Parameters from plugin config
            global_params: Parameters passed to run/fit

        Returns:
            Merged parameter dictionary
        """
        # Start with defaults
        merged = plugin_cls.get_default_params().copy()

        # Override with config params
        merged.update(config_params)

        # Override with global params
        merged.update(global_params)

        return merged

    def validate(self) -> Dict[str, dict]:
        """
        Validate all plugins in the pipeline.

        Returns:
            Dictionary mapping plugin name to {valid: bool, error: str}
        """
        self._initialize()

        results: Dict[str, dict] = {}
        for plugin_config, instance in self._execution_order:
            try:
                is_valid = instance.validate()
                results[plugin_config.name] = {
                    "valid": is_valid,
                    "error": "" if is_valid else "Validation returned False",
                }
            except Exception as e:
                results[plugin_config.name] = {
                    "valid": False,
                    "error": str(e),
                }

        return results

    def fit(self, ctx: PipelineContext, **global_params: Any) -> None:
        """
        Fit stateful plugins on training data.

        Args:
            ctx: Pipeline context with training data
            **global_params: Parameters passed to all plugins
        """
        self._initialize()

        for plugin_config, instance in self._execution_order:
            if instance.stateful:
                plugin_cls = type(instance)
                merged_params = self._merge_params(
                    plugin_cls, plugin_config.params, global_params
                )
                instance.fit(ctx, **merged_params)

    def run(
        self,
        ctx: PipelineContext,
        phases: Optional[List[str]] = None,
        **global_params: Any,
    ) -> PipelineContext:
        """
        Execute the pipeline on the given context.

        Args:
            ctx: Pipeline context to process
            phases: Optional list of phase names to run (default: all phases)
            **global_params: Parameters passed to all plugins

        Returns:
            Updated pipeline context
        """
        self._initialize()

        # Determine which phases to run
        if phases is None:
            allowed_phases = {phase_name for phase_name, _ in self.PHASE_ORDER}
        else:
            allowed_phases = set(phases)

        # Execute plugins in order
        for plugin_config, instance in self._execution_order:
            # Check if this plugin's phase should be run
            plugin_cls = type(instance)
            phase_name = None
            for pn, pe in self.PHASE_ORDER:
                if pe == plugin_cls.phase:
                    phase_name = pn
                    break

            if phase_name not in allowed_phases:
                continue

            # Merge parameters
            merged_params = self._merge_params(
                plugin_cls, plugin_config.params, global_params
            )

            # Execute plugin
            ctx = instance.execute(ctx, **merged_params)

        return ctx

    def reset(self) -> None:
        """Reset all stateful plugins to their initial unfitted state."""
        for _plugin_config, instance in self._execution_order:
            if instance.stateful:
                instance.reset()

    def get_instance(self, name: str) -> Optional[BasePlugin]:
        """
        Get a plugin instance by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None if not found
        """
        self._initialize()
        return self._instances.get(name)
