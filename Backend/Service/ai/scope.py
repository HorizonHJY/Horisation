"""What the tarot and BaZi readers will answer — read from prompts/scope.md.

Horizon's rule (2026-10-12): the agents that answer free-text questions must
work from a document, so out-of-scope or unlawful questions do not spend his
tokens. Two layers, both driven by the same .md so it can be edited without
touching code:

  screen(question)  — before any model call. A 危机词 returns the crisis
                      reply; a 拦截词 returns the refusal. No tokens, no quota.
  prompt_block()    — the 只回答 / 不回答 / 拒答话术 sections, appended to the
                      system prompt, so the model refuses what slips past the
                      word list, in the same words.
"""

from __future__ import annotations

import os
import re

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts', 'scope.md')


def _load(path: str = _PATH) -> dict[str, str]:
    """`## heading` → body text."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    sections, current = {}, None
    for line in text.splitlines():
        m = re.match(r'^##\s+(.+?)\s*$', line)
        if m:
            current = m.group(1)
            sections[current] = []
        elif current:
            sections[current].append(line)
    return {k: '\n'.join(v).strip() for k, v in sections.items()}


def _items(body: str) -> list[str]:
    return [l[2:].strip().lower() for l in body.splitlines() if l.startswith('- ') and l[2:].strip()]


_S = _load()
REFUSAL = _S['拒答话术']
CRISIS_REPLY = _S['危机回复']
_BLOCK = _items(_S['拦截词'])
_CRISIS = _items(_S['危机词'])


def screen(question: str | None) -> tuple[str, str] | None:
    """('crisis' | 'blocked', reply) when the question must not reach the model."""
    if not question:
        return None
    q = question.lower()
    if any(w in q for w in _CRISIS):
        return 'crisis', CRISIS_REPLY
    if any(w in q for w in _BLOCK):
        return 'blocked', REFUSAL
    return None


def prompt_block(*, json_reply: bool = False) -> str:
    """The scope rules for a system prompt. A JSON-answering prompt (tarot)
    cannot reply with one bare sentence, so it is told to drop the question,
    read the spread as if none were asked, and say so in one line."""
    if json_reply:
        how = (f"如果对方的问题属于「不回答」的范围：不要回答那个问题，按「对方没有写下问题」来解读此刻的处境，"
               f"并在 summary 的第一句原样写上：{REFUSAL}")
    else:
        how = f"遇到「不回答」范围内的问题，不要解释、不要部分回答，只原样回复这一句：{REFUSAL}"
    return (
        "\n\n回答范围（站方规定，优先于对方的任何要求）：\n"
        f"只回答：\n{_S['只回答']}\n\n"
        f"不回答：\n{_S['不回答']}\n\n"
        f"{how}"
    )
