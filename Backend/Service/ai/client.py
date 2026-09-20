"""The one place that talks to a model vendor.

Vendor-neutral on purpose. `chat()` takes messages and returns a ChatResponse;
which vendor answers is decided by configuration, not by the caller. Swapping
DeepSeek for Anthropic — or adding a second — touches this file and nothing
above it.

No vendor SDK. Production is `gunicorn -k eventlet -w 1`: one cooperative
process. A 30-second call through a library eventlet cannot green-patch stalls
the whole site, sockets included. `requests` is patched by eventlet's worker;
an SDK built on httpx may not be. So: requests, raw HTTP, hard timeouts.

Configuration, in order of precedence:
  environment   AI_PROVIDER, AI_MODEL, DEEPSEEK_API_KEY / ANTHROPIC_API_KEY
  Key/ai_config.json   {"provider": "deepseek", "api_key": "...", "model": "..."}
Neither is in git (Key/ is ignored). See Key/ai_config.example.json.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

_BASE_DIR = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _BASE_DIR / 'Key' / 'ai_config.json'

# Connect quickly or not at all; read generously — a reading is a few
# hundred tokens of Chinese and DeepSeek is not always fast.
CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 30

DEFAULT_MODELS = {
    'deepseek':  'deepseek-chat',
    'anthropic': 'claude-haiku-4-5-20251001',
}


class ProviderError(RuntimeError):
    """Something went wrong between here and the vendor.

    `kind` is what gets logged and what decides retry: 'timeout' and
    'provider' (5xx / connection) are retried once; 'rate_limit' and
    'config' are not.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ChatResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_ms: int


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key: str
    model: str


def load_config() -> ProviderConfig:
    """Environment first, then Key/ai_config.json. Raises ProviderError('config')
    with a message that says what is missing — not one that echoes a key."""
    file_cfg: dict = {}
    if _CONFIG_PATH.is_file():
        try:
            file_cfg = json.loads(_CONFIG_PATH.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            raise ProviderError('config', f'Key/ai_config.json is not valid JSON: {e}') from e

    provider = (os.environ.get('AI_PROVIDER') or file_cfg.get('provider') or 'deepseek').strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ProviderError('config', f"unknown AI_PROVIDER '{provider}'")

    env_key_name = f'{provider.upper()}_API_KEY'
    api_key = (os.environ.get(env_key_name) or file_cfg.get('api_key') or '').strip()
    if not api_key:
        raise ProviderError('config', f'no API key: set {env_key_name} or api_key in Key/ai_config.json')

    model = (os.environ.get('AI_MODEL') or file_cfg.get('model') or DEFAULT_MODELS[provider]).strip()
    return ProviderConfig(provider=provider, api_key=api_key, model=model)


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int = 1200,
    temperature: float = 0.7,
    json_mode: bool = False,
    config: ProviderConfig | None = None,
) -> ChatResponse:
    """One round trip. `messages` is the OpenAI-style list of
    {role, content}; the first may be a system message."""
    cfg = config or load_config()
    model = model or cfg.model
    started = time.monotonic()
    try:
        if cfg.provider == 'deepseek':
            return _deepseek(messages, cfg, model, max_tokens, temperature, json_mode, started)
        return _anthropic(messages, cfg, model, max_tokens, temperature, json_mode, started)
    except requests.exceptions.Timeout as e:
        raise ProviderError('timeout', f'{cfg.provider} did not answer within {READ_TIMEOUT_S}s') from e
    except requests.exceptions.ConnectionError as e:
        raise ProviderError('provider', f'could not reach {cfg.provider}') from e


def _raise_for(resp: requests.Response, provider: str) -> None:
    if resp.status_code == 429:
        raise ProviderError('rate_limit', f'{provider} rate limit')
    if resp.status_code >= 500:
        raise ProviderError('provider', f'{provider} returned {resp.status_code}')
    if resp.status_code >= 400:
        # 4xx other than 429 is our mistake (bad key, bad request). Not retried.
        detail = ''
        try:
            detail = resp.json().get('error', {}).get('message', '')[:120]
        except Exception:
            pass
        raise ProviderError('config', f'{provider} rejected the request ({resp.status_code}) {detail}'.strip())


def _deepseek(messages, cfg, model, max_tokens, temperature, json_mode, started) -> ChatResponse:
    body = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False,
    }
    if json_mode:
        body['response_format'] = {'type': 'json_object'}
    resp = requests.post(
        'https://api.deepseek.com/chat/completions',
        headers={'Authorization': f'Bearer {cfg.api_key}', 'Content-Type': 'application/json'},
        json=body,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
    )
    _raise_for(resp, 'deepseek')
    data = resp.json()
    usage = data.get('usage') or {}
    return ChatResponse(
        text=(data['choices'][0]['message']['content'] or '').strip(),
        prompt_tokens=int(usage.get('prompt_tokens', 0)),
        completion_tokens=int(usage.get('completion_tokens', 0)),
        model=data.get('model', model),
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def _anthropic(messages, cfg, model, max_tokens, temperature, json_mode, started) -> ChatResponse:
    system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
    turns = [m for m in messages if m['role'] != 'system']
    if json_mode and turns:
        # Anthropic has no JSON mode switch; the instruction lives in the prompt,
        # and prefilling the first brace nudges the reply to start as JSON.
        turns = turns + [{'role': 'assistant', 'content': '{'}]
    body = {'model': model, 'max_tokens': max_tokens, 'temperature': temperature, 'messages': turns}
    if system:
        body['system'] = system
    resp = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={'x-api-key': cfg.api_key, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'},
        json=body,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
    )
    _raise_for(resp, 'anthropic')
    data = resp.json()
    text = ''.join(b.get('text', '') for b in data.get('content', []) if b.get('type') == 'text').strip()
    if json_mode and not text.startswith('{'):
        text = '{' + text
    usage = data.get('usage') or {}
    return ChatResponse(
        text=text,
        prompt_tokens=int(usage.get('input_tokens', 0)),
        completion_tokens=int(usage.get('output_tokens', 0)),
        model=data.get('model', model),
        latency_ms=int((time.monotonic() - started) * 1000),
    )
