"""Cross-service storage contract required by fwbg-agents DSR accounting."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_agents_share_fwbg_workspace_and_test_results_path() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose["services"]
    agents = services["agents"]

    agents_workspace = next(
        volume
        for volume in agents["volumes"]
        if isinstance(volume, dict) and volume.get("target") == "/root/fwbg"
    )
    assert agents_workspace == {
        "type": "volume",
        "source": "fwbg-workspace",
        "target": "/root/fwbg",
        "read_only": True,
    }
    assert "fwbg-workspace:/root/fwbg" in services["api"]["volumes"]
    assert "FWBG_TEST_RESULTS_DIR=/root/fwbg/test_results" in agents["environment"]
