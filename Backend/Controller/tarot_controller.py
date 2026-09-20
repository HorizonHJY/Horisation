"""
tarot_controller.py
Three-card tarot spread — past, present, future — and its reading.

The deck itself is static data, but the draw happens here rather than in the
browser: a reading whose outcome can be inspected or re-rolled from devtools is
not a reading. `secrets` rather than `random` for the same reason — this is the
one thing the feature is for, so it should not be a predictable PRNG.

The same rule reaches the AI reading. /draw records the spread and hands back
a reading_id; /reading takes only that id. The three cards the model is told
about are the three the server drew, never three the client sent — or you
could change the spread and then ask.

Upright only for now; the deck file carries no reversed meanings.

Assets:
  images  metabismuth/tarot-json (MIT) — scans of the Rider-Waite-Smith deck,
          public domain in the US, not in the EU
  text    A. E. Waite, "The Pictorial Key to the Tarot" (1911), public domain
"""

import json
import os
import re
import secrets

from flask import Blueprint, jsonify, request

from Backend.Controller.auth_controller import login_required
from Backend.Controller import tarot_db
from Backend.Service.ai import ai, today_key
from Backend.Service.ai.prompts.tarot import MAX_QUESTION_CHARS

tarot_bp = Blueprint('tarot', __name__, url_prefix='/api/tarot')

POSITIONS = [
    {'key': 'past',    'label': 'Past',    'label_zh': '过去'},
    {'key': 'present', 'label': 'Present', 'label_zh': '现在'},
    {'key': 'future',  'label': 'Future',  'label_zh': '未来'},
]

_DECK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tarot_deck.json')

_deck_cache = None

# Control characters other than newline/tab have no place in a question.
_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _deck() -> list:
    """Load the deck once and keep it — 78 static rows, read on every draw."""
    global _deck_cache
    if _deck_cache is None:
        with open(_DECK_PATH, encoding='utf-8') as f:
            _deck_cache = json.load(f)
    return _deck_cache


def _me() -> tuple[str, str | None]:
    u = request.current_user
    return u['username'], u.get('role')


@tarot_bp.route('/deck', methods=['GET'])
@login_required
def get_deck():
    """The whole deck, for laying out the fan of face-down cards."""
    deck = _deck()
    return jsonify({
        'ok': True,
        'count': len(deck),
        # The fan only needs identity and the image path; meanings ride along
        # so a drawn card never has to be fetched a second time.
        'cards': deck,
        'positions': POSITIONS,
    })


@tarot_bp.route('/draw', methods=['POST'])
@login_required
def draw():
    """Draw three distinct cards, one per position, and remember them.

    Distinct because a spread with the same card twice is not a spread. The
    client is told which cards came up and where; it is never told the order of
    the rest of the deck, so nothing about the next draw leaks.

    Drawing is free and unmetered. The reading is what is rationed.
    """
    deck = _deck()
    if len(deck) < len(POSITIONS):
        return jsonify({'ok': False, 'error': 'Deck is incomplete.'}), 500

    picked = []
    seen = set()
    while len(picked) < len(POSITIONS):
        card = deck[secrets.randbelow(len(deck))]
        if card['id'] in seen:
            continue
        seen.add(card['id'])
        picked.append(card)

    spread = [{'position': POSITIONS[i], 'card': picked[i]} for i in range(len(POSITIONS))]
    username, _ = _me()
    reading_id = tarot_db.create_reading(username, spread)

    return jsonify({'ok': True, 'spread': spread, 'reading_id': reading_id})


@tarot_bp.route('/reading', methods=['POST'])
@login_required
def reading():
    """Interpret a spread the server drew.

    Body: { reading_id, question? }. A reading that already exists is returned
    as it is — a second click does not cost a second call.
    """
    body = request.get_json(silent=True) or {}
    rid = str(body.get('reading_id') or '').strip()
    if not rid:
        return jsonify({'ok': False, 'error': 'reading_id is required'}), 400

    username, role = _me()
    row = tarot_db.get_reading(rid, username)
    if row is None:
        return jsonify({'ok': False, 'error': 'No such spread.'}), 404

    if row.read_at is not None:
        return jsonify({'ok': True, 'reading': row.to_dict(include_spread=False), 'cached': True})

    question = body.get('question')
    if question is not None:
        question = _CONTROL.sub('', str(question)).strip()
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({'ok': False, 'error': f'Question is too long (max {MAX_QUESTION_CHARS} characters).'}), 400
        if not question:
            question = None

    spread = json.loads(row.spread_json)
    result = ai.run(
        feature='tarot',
        user=username,
        role=role,
        quota_key=f'user:{username}:tarot:{today_key()}',
        payload={'spread': spread, 'question': question},
    )

    if not result.ok:
        status = {'quota': 429, 'global_cap': 503, 'disabled': 503, 'config': 503}.get(result.error_kind, 502)
        return jsonify({
            'ok': False,
            'error': result.error,
            'error_kind': result.error_kind,
            # Failures that did not consume a turn say so, so the page can offer a retry.
            'retryable': result.error_kind in ('timeout', 'provider', 'rate_limit'),
            'quota': result.quota.as_dict() if result.quota else None,
        }), status

    tarot_db.save_interpretation(
        rid, question=question, raw=result.text, data=result.data,
        model=result.usage.model, prompt_version=result.prompt_version,
    )
    row = tarot_db.get_reading(rid, username)
    return jsonify({
        'ok': True,
        'reading': row.to_dict(include_spread=False),
        'quota': result.quota.as_dict(),
        'cached': False,
    })


@tarot_bp.route('/readings/<rid>/rating', methods=['POST'])
@login_required
def rate(rid):
    """How well did it fit — 1 to 5, and an optional line. Yours only; can be changed."""
    body = request.get_json(silent=True) or {}
    try:
        rating = int(body.get('rating'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'rating must be 1–5'}), 400
    if not 1 <= rating <= 5:
        return jsonify({'ok': False, 'error': 'rating must be 1–5'}), 400
    note = body.get('note')
    if note is not None:
        note = _CONTROL.sub('', str(note)).strip()[:100] or None

    username, _ = _me()
    if not tarot_db.save_rating(rid, username, rating, note):
        return jsonify({'ok': False, 'error': 'No such reading.'}), 404
    return jsonify({'ok': True, 'rating': rating, 'note': note})


@tarot_bp.route('/readings', methods=['GET'])
@login_required
def history():
    """Your own interpreted readings, newest first. ?before=<iso> pages back."""
    username, _ = _me()
    before = request.args.get('before') or None
    try:
        limit = max(1, min(50, int(request.args.get('limit', 20))))
    except ValueError:
        limit = 20
    return jsonify({'ok': True, 'readings': tarot_db.list_readings(username, limit=limit, before=before)})
