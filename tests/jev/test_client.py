from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.jev.client import JevProvider, ask, parse_response

OFFICIAL = JevProvider(
    name="jev_official",
    base_url="https://api.example-jev.invalid/v1/decide",
    api_key_env="JEV_OFFICIAL_API_KEY",
)


def test_ask_builds_correct_request(monkeypatch):
    monkeypatch.setenv("JEV_OFFICIAL_API_KEY", "test-key")
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "results": {
            "is_long_win": {"probability": 0.62},
            "is_short_win": {"probability": 0.15},
        }
    }
    fake_response.raise_for_status.return_value = None

    with patch("scripts.jev.client.requests.post", return_value=fake_response) as post:
        result = ask(
            OFFICIAL,
            state="EURUSD @ 1.0850, RSI(14)=61, ...",
            questions={
                "is_long_win": {
                    "type": "noul",
                    "instructions": "Will TP be hit before SL/timeout for a long entry here?",
                    "criteria": {"true": "TP hit first", "false": "SL hit first or timeout"},
                },
                "is_short_win": {
                    "type": "noul",
                    "instructions": "Will TP be hit before SL/timeout for a short entry here?",
                    "criteria": {"true": "TP hit first", "false": "SL hit first or timeout"},
                },
            },
        )

    called_url, called_kwargs = post.call_args[0][0], post.call_args[1]
    assert called_url == OFFICIAL.base_url
    assert called_kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert called_kwargs["json"]["state"].startswith("EURUSD")
    assert set(called_kwargs["json"]["questions"]) == {"is_long_win", "is_short_win"}
    assert result == {"is_long_win": 0.62, "is_short_win": 0.15}


def test_parse_response_missing_question_raises():
    with pytest.raises(KeyError):
        parse_response(
            {"results": {"is_long_win": {"probability": 0.5}}}, ["is_long_win", "is_short_win"]
        )


def test_ask_missing_api_key_env_raises(monkeypatch):
    monkeypatch.delenv("JEV_OFFICIAL_API_KEY", raising=False)

    with pytest.raises(KeyError):
        ask(OFFICIAL, state="EURUSD @ 1.0850", questions={"is_long_win": {}})


@pytest.mark.parametrize("bad_probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_parse_response_rejects_invalid_probability(bad_probability):
    with pytest.raises(ValueError):
        parse_response(
            {"results": {"is_long_win": {"probability": bad_probability}}}, ["is_long_win"]
        )


def test_ask_raises_on_http_error(monkeypatch):
    monkeypatch.setenv("JEV_OFFICIAL_API_KEY", "test-key")
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

    with patch("scripts.jev.client.requests.post", return_value=fake_response):
        with pytest.raises(requests.HTTPError):
            ask(OFFICIAL, state="EURUSD @ 1.0850", questions={"is_long_win": {}})
