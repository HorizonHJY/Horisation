"""The registry: feature name → Feature.

A Feature is the whole of what the AI layer needs to know about one use:
how to turn a payload into messages, how to read the reply back, and what
generation settings it wants. Everything else — key, vendor, quota, retry,
cost — is the layer's, and identical for every feature.

To add a feature: one module in prompts/, one Feature here. Nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..prompts import tarot as tarot_prompt


@dataclass(frozen=True)
class Feature:
    name: str
    prompt_version: str
    build: Callable[[dict], list[dict]]        # payload → messages
    parse: Callable[[str], dict | None]        # reply text → data, or None
    json_mode: bool = True
    max_tokens: int = 1200
    temperature: float = 0.7


def _tarot_build(payload: dict) -> list[dict]:
    return tarot_prompt.build(payload['spread'], payload.get('question'))


FEATURES: dict[str, Feature] = {
    'tarot': Feature(
        name='tarot',
        prompt_version=tarot_prompt.PROMPT_VERSION,
        build=_tarot_build,
        parse=tarot_prompt.parse,
        json_mode=True,
        max_tokens=1200,
        temperature=0.75,
    ),
}


def get(name: str) -> Feature:
    try:
        return FEATURES[name]
    except KeyError:
        raise KeyError(f"unknown AI feature '{name}'. Registered: {', '.join(FEATURES)}") from None
