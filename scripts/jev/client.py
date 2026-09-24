"""Thin HTTP client for Jev-shaped decision APIs (state + questions -> probabilities).

ASSUMPTION (unverified against a real API as of 2026-09-24): the response is
JSON shaped as {"results": {"<question_name>": {"probability": <float>, ...}}}.
If a real call comes back differently, fix parse_response() only — every
other module in scripts/jev/ depends on ask()'s return value
(dict[question_name] -> float), not on the raw response shape.
"""

import math
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
    """Extract each requested probability from a Jev-shaped response."""
    results = raw["results"]
    parsed = {}
    for name in question_names:
        probability = float(results[name]["probability"])
        if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
            raise ValueError(f"invalid probability for {name!r}: {probability!r}")
        parsed[name] = probability
    return parsed


def ask(provider: JevProvider, state: str, questions: dict) -> dict[str, float]:
    """Submit a state and questions using the provider's configured credentials."""
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
