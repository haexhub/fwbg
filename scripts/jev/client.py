"""Thin HTTP client for Jev-shaped decision APIs (state + questions -> probabilities).

ASSUMPTION (unverified against a real API as of 2026-09-24): the response is
JSON shaped as {"results": {"<question_name>": {"probability": <float>, ...}}}.
If a real call comes back differently, fix parse_response() only — every
other module in scripts/jev/ depends on ask()'s return value
(dict[question_name] -> float), not on the raw response shape.
"""

import os
from dataclasses import dataclass, field

import requests


@dataclass(frozen=True)
class JevProvider:
    name: str
    base_url: str
    api_key_env: str
    extra_headers: dict = field(default_factory=dict)
    timeout_s: float = 10.0


def parse_response(raw: dict, question_names: list[str]) -> dict[str, float]:
    results = raw["results"]
    return {name: float(results[name]["probability"]) for name in question_names}


def ask(provider: JevProvider, state: str, questions: dict) -> dict[str, float]:
    api_key = os.environ[provider.api_key_env]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **provider.extra_headers,
    }
    response = requests.post(
        provider.base_url,
        headers=headers,
        json={"state": state, "questions": questions},
        timeout=provider.timeout_s,
    )
    response.raise_for_status()
    return parse_response(response.json(), list(questions.keys()))
