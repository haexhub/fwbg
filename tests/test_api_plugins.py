"""Regression tests for plugin metadata error handling."""

import pytest
from fastapi import HTTPException

import fwbg.api.plugins as plugins
from fwbg.pipeline.registry import PluginNotFoundError


def test_list_plugins_skips_plugin_with_broken_metadata(monkeypatch):
    class Registry:
        def list_plugins(self, **kwargs):
            return ["fwbg-core:broken", "fwbg-core:healthy"]

    def plugin_to_dict(fqn):
        if fqn.endswith(":broken"):
            raise RuntimeError("broken metadata")
        return {"fqn": fqn}

    monkeypatch.setattr(plugins, "get_plugin_registry", lambda: Registry())
    monkeypatch.setattr(plugins, "_plugin_to_dict", plugin_to_dict)

    assert plugins.list_plugins(phase=None, namespace=None) == [
        {
            "fqn": "fwbg-core:broken",
            "name": "broken",
            "namespace": "fwbg-core",
            "broken": True,
            "metadata_error": "broken metadata",
        },
        {"fqn": "fwbg-core:healthy"},
    ]


def test_get_plugin_returns_404_for_lookup_failure(monkeypatch):
    class Registry:
        def get(self, fqn):
            raise PluginNotFoundError(fqn)

    monkeypatch.setattr(plugins, "get_plugin_registry", lambda: Registry())

    with pytest.raises(HTTPException) as exc_info:
        plugins.get_plugin("fwbg-core:missing")

    assert exc_info.value.status_code == 404


def test_get_plugin_does_not_hide_metadata_errors(monkeypatch):
    class Registry:
        def get(self, fqn):
            return object()

    monkeypatch.setattr(plugins, "get_plugin_registry", lambda: Registry())
    monkeypatch.setattr(
        plugins,
        "_plugin_to_dict",
        lambda fqn: (_ for _ in ()).throw(RuntimeError("broken metadata")),
    )

    with pytest.raises(RuntimeError, match="broken metadata"):
        plugins.get_plugin("fwbg-core:broken")
