"""
tarot_controller.py
Three-card tarot spread — past, present, future — and its reading.

The deck itself is static data, but the draw happens here rather than in the
browser: a reading whose outcome can be inspected or re-rolled from devtools is
not a reading. `secrets` rather than `random` for the same reason — this is the
one thing the feature is for, so it should not be a predictable PRNG.

The card you touch is the card you get. /draw shuffles the whole deck on the
server and keeps the order there; the fan in the browser is laid out in the
static deck order, so a seat says nothing about which card sits under it.
/pick turns over the card at the seat you touched. The choice is real, and
still nobody can see or steer it from devtools.

The same rule reaches the AI reading. /draw hands back a reading_id; /reading
takes only that id. The three cards the model is told
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
import logging
import secrets

from flask import Blueprint, jsonify, request

from Backend.Controller.auth_controller import login_required, login_optional
from Backend.Controller import tarot_db
from Backend.Controller.socketio_instance import socketio
from Backend.Service.ai import ai, today_key
from Backend.Service.ai.prompts.tarot import MAX_FOLLOWUP_CHARS, MAX_QUESTION_CHARS
from Backend.Service.ai import scope

log = logging.getLogger(__name__)

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
@login_optional
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
    """Shuffle the deck and remember the order. Nothing is turned over yet.

    The client is never told the order — only a reading_id. Cards come out
    one at a time through /pick, at the seats the reader touches.

    Drawing is free and unmetered. The reading is what is rationed.
    """
    deck = _deck()
    if len(deck) < len(POSITIONS):
        return jsonify({'ok': False, 'error': 'Deck is incomplete.'}), 500

    order = [c['id'] for c in deck]
    secrets.SystemRandom().shuffle(order)
    username, _ = _me()
    reading_id = tarot_db.create_reading(username, order)

    return jsonify({'ok': True, 'reading_id': reading_id})


@tarot_bp.route('/pick', methods=['POST'])
@login_required
def pick():
    """Turn over the card at one seat of the shuffled fan.

    Body: { reading_id, seat } — seat is 0..77, the card's place in the fan.
    The card fills the next empty position (past, then present, then future).
    """
    body = request.get_json(silent=True) or {}
    rid = str(body.get('reading_id') or '').strip()
    seat = body.get('seat')
    if not rid or not isinstance(seat, int) or isinstance(seat, bool):
        return jsonify({'ok': False, 'error': 'reading_id and seat are required'}), 400

    username, _ = _me()
    row = tarot_db.get_reading(rid, username)
    if row is None or not row.deck_order_json:
        return jsonify({'ok': False, 'error': 'No such spread.'}), 404

    order = json.loads(row.deck_order_json)
    spread = json.loads(row.spread_json)
    if not 0 <= seat < len(order):
        return jsonify({'ok': False, 'error': 'No card at that seat.'}), 400
    if len(spread) >= len(POSITIONS):
        return jsonify({'ok': False, 'error': 'All three cards are already taken.'}), 409
    if any(e.get('seat') == seat for e in spread):
        return jsonify({'ok': False, 'error': 'That card is already taken.'}), 409

    by_id = {c['id']: c for c in _deck()}
    entry = {'position': POSITIONS[len(spread)], 'card': by_id[order[seat]], 'seat': seat}
    if not tarot_db.append_pick(rid, row.spread_json, spread + [entry]):
        return jsonify({'ok': False, 'error': 'Another pick landed first. Try again.'}), 409

    return jsonify({'ok': True, 'pick': entry, 'done': len(spread) + 1 == len(POSITIONS)})


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

    spread = json.loads(row.spread_json)
    if len(spread) < len(POSITIONS):
        return jsonify({'ok': False, 'error': 'Take all three cards first.'}), 409

    question = body.get('question')
    if question is not None:
        question = _CONTROL.sub('', str(question)).strip()
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({'ok': False, 'error': f'Question is too long (max {MAX_QUESTION_CHARS} characters).'}), 400
        if not question:
            question = None
        # Out of scope or unlawful: answered here, before the model — no
        # tokens, no turn spent. The rules live in prompts/scope.md.
        screened = scope.screen(question)
        if screened:
            kind, reply = screened
            return jsonify({'ok': False, 'error': reply, 'error_kind': kind, 'retryable': False}), 422

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


# ── Follow-up questions ──────────────────────────────────────────────────────

def _run_followup(fid: str, username: str, role: str | None, payload: dict) -> None:
    try:
        result = ai.run(
            feature='tarot_ask', user=username, role=role,
            quota_key=f'user:{username}:tarot_ask:{today_key()}',
            payload=payload,
        )
    except Exception:
        log.exception('tarot follow-up %s crashed', fid)
        tarot_db.fail_followup(fid, {'kind': 'crash', 'error': '没有答上来，请换个说法再问一次。'})
        return
    if result.ok:
        tarot_db.answer(fid, text_=result.data['answer'], model=result.usage.model,
                        prompt_version=result.prompt_version)
    else:
        tarot_db.fail_followup(fid, {'kind': result.error_kind,
                                     'error': result.error or '没有答上来，请换个说法再问一次。'})


@tarot_bp.route('/readings/<rid>/ask', methods=['POST'])
@login_required
def ask(rid):
    """Body: { question }. A question about a spread already read.
    Answered in the background; poll GET /followups/<id>."""
    q = _CONTROL.sub('', str((request.get_json(silent=True) or {}).get('question') or '')).strip()
    if not q:
        return jsonify({'ok': False, 'error': '请写下你的问题。'}), 400
    if len(q) > MAX_FOLLOWUP_CHARS:
        return jsonify({'ok': False, 'error': f'问题太长了（最多 {MAX_FOLLOWUP_CHARS} 字）。'}), 400

    username, role = _me()
    row = tarot_db.get_reading(rid, username)
    if row is None:
        return jsonify({'ok': False, 'error': '没有这次占卜。'}), 404
    if row.read_at is None:
        return jsonify({'ok': False, 'error': '先完成整体解读，再追问。'}), 409

    screened = scope.screen(q)
    if screened:
        f = tarot_db.ask_screened(rid, username, q, screened[1], kind=screened[0])
        return jsonify({'ok': True, 'followup': f.to_dict()})

    history = [(f.question, f.answer) for f in tarot_db.followups(rid) if f.answer]
    payload = {
        'spread': json.loads(row.spread_json),
        'reading': json.loads(row.reading_json) if row.reading_json else None,
        'reading_text': row.reading_raw,
        'history': history,
        'question': q,
    }
    f = tarot_db.ask(rid, username, q)
    socketio.start_background_task(_run_followup, f.id, username, role, payload)
    return jsonify({'ok': True, 'followup': f.to_dict()})


@tarot_bp.route('/followups/<fid>', methods=['GET'])
@login_required
def poll_followup(fid):
    username, _ = _me()
    f = tarot_db.get_followup(fid, username)
    if f is None:
        return jsonify({'ok': False, 'error': '没有这条提问。'}), 404
    return jsonify({'ok': True, 'followup': f.to_dict()})



@tarot_bp.route('/readings/<rid>/followups', methods=['GET'])
@login_required
def list_followups(rid):
    """The questions already asked about this spread, oldest first."""
    username, _ = _me()
    if tarot_db.get_reading(rid, username) is None:
        return jsonify({'ok': False, 'error': '没有这次占卜。'}), 404
    return jsonify({'ok': True, 'followups': [f.to_dict() for f in tarot_db.followups(rid)]})

