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

from ..prompts import bazi as bazi_prompt
from ..prompts import bazi_compat as compat_prompt
from ..prompts import listing_autofill as listing_autofill_prompt
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
    read_timeout: int = 30                     # seconds; a long answer needs longer
    # Override the configured text model when this feature needs a different
    # one (today: vision reads of listing photos go to DeepSeek-VL2 while
    # the rest of the site stays on deepseek-chat). None = inherit.
    model_override: str | None = None


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
    # A full chart reading is ~2,500 tokens of JSON — several times a tarot
    # reading — so it gets the room and the wait that takes.
    'bazi': Feature(
        name='bazi',
        prompt_version=bazi_prompt.PROMPT_VERSION,
        build=lambda payload: bazi_prompt.build(payload['chart']),
        parse=bazi_prompt.parse,
        json_mode=True,
        max_tokens=3500,
        temperature=0.6,
        read_timeout=120,
    ),
    # A question about a chart already read: short plain-text answer.
    'bazi_ask': Feature(
        name='bazi_ask',
        prompt_version=bazi_prompt.FOLLOWUP_VERSION,
        build=bazi_prompt.build_followup,
        parse=bazi_prompt.parse_followup,
        json_mode=False,
        max_tokens=800,
        temperature=0.6,
        read_timeout=60,
    ),
    # 合盘 reading: two charts in, structured JSON out. Longer than a single
    # chart reading because the model is comparing two tables.
    'bazi_compat': Feature(
        name='bazi_compat',
        prompt_version=compat_prompt.PROMPT_VERSION,
        build=compat_prompt.build,
        parse=compat_prompt.parse,
        json_mode=True,
        max_tokens=3500,
        temperature=0.6,
        read_timeout=120,
    ),
    # A question after the reading: plain text, short, conversational.
    'tarot_ask': Feature(
        name='tarot_ask',
        prompt_version=tarot_prompt.FOLLOWUP_VERSION,
        build=tarot_prompt.build_followup,
        parse=tarot_prompt.parse_followup,
        json_mode=False,
        max_tokens=700,
        temperature=0.75,
        read_timeout=45,
    ),
    # The home-page daily card: one card already chosen by the server, two
    # short sentences in JSON. Tiny by design — short timeout, short budget.
    'tarot_daily': Feature(
        name='tarot_daily',
        prompt_version=tarot_prompt.DAILY_VERSION,
        build=lambda payload: tarot_prompt.build_daily(payload['card'], payload['upright']),
        parse=tarot_prompt.parse_daily,
        json_mode=True,
        max_tokens=300,
        temperature=0.7,
        read_timeout=20,
    ),
    'bazi_compat_ask': Feature(
        name='bazi_compat_ask',
        prompt_version=compat_prompt.FOLLOWUP_VERSION,
        build=compat_prompt.build_followup,
        parse=compat_prompt.parse_followup,
        json_mode=False,
        max_tokens=800,
        temperature=0.6,
        read_timeout=60,
    ),
    # Vision read of one listing photo. deepseek-flash is a thinking model
    # that eats ~700 reasoning tokens before writing its answer, so this
    # budget has to be big (600 was all reasoning and empty content, 2500
    # leaves room for a JSON draft). Verified on real listing photos:
    # about 6 seconds per call, ~$0.0005.
    'listing_autofill': Feature(
        name='listing_autofill',
        prompt_version=listing_autofill_prompt.PROMPT_VERSION,
        build=lambda payload: listing_autofill_prompt.build(payload['image_data_url']),
        parse=listing_autofill_prompt.parse,
        json_mode=True,
        max_tokens=2500,
        temperature=0.3,
        read_timeout=45,
        model_override='deepseek-flash',
    ),
}


def get(name: str) -> Feature:
    try:
        return FEATURES[name]
    except KeyError:
        raise KeyError(f"unknown AI feature '{name}'. Registered: {', '.join(FEATURES)}") from None
