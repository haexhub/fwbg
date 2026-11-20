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
    assert "fwbg-workspace:/home/fwbg/fwbg" in services["api"]["volumes"]
    assert "FWBG_TEST_RESULTS_DIR=/root/fwbg/test_results" in agents["environment"]


def test_fwbg_image_runs_unprivileged_and_mounts_under_its_home() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text().splitlines()
    assert "USER fwbg:fwbg" in dockerfile
    assert any("--uid 10001" in line for line in dockerfile)

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    for name in ("ig-bot", "api"):
        service = compose["services"][name]
        assert "user" not in service
        for volume in service["volumes"]:
            target = (
                volume["target"] if isinstance(volume, dict) else volume.split(":")[1]
            )
            assert not target.startswith("/root"), (name, target)
