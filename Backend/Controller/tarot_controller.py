"""
tarot_controller.py
Three-card tarot spread — past, present, future.

The deck itself is static data, but the draw happens here rather than in the
browser: a reading whose outcome can be inspected or re-rolled from devtools is
not a reading. `secrets` rather than `random` for the same reason — this is the
one thing the feature is for, so it should not be a predictable PRNG.

Upright only for now; the deck file carries no reversed meanings.

Assets:
  images  metabismuth/tarot-json (MIT) — scans of the Rider-Waite-Smith deck,
          public domain in the US, not in the EU
  text    A. E. Waite, "The Pictorial Key to the Tarot" (1911), public domain
"""

import json
import os
import secrets

from flask import Blueprint, jsonify

from Backend.Controller.auth_controller import login_required

tarot_bp = Blueprint('tarot', __name__, url_prefix='/api/tarot')

POSITIONS = [
    {'key': 'past',    'label': 'Past',    'label_zh': '过去'},
    {'key': 'present', 'label': 'Present', 'label_zh': '现在'},
    {'key': 'future',  'label': 'Future',  'label_zh': '未来'},
]

_DECK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tarot_deck.json')

_deck_cache = None


def _deck() -> list:
    """Load the deck once and keep it — 78 static rows, read on every draw."""
    global _deck_cache
    if _deck_cache is None:
        with open(_DECK_PATH, encoding='utf-8') as f:
            _deck_cache = json.load(f)
    return _deck_cache


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
    """Draw three distinct cards, one per position.

    Distinct because a spread with the same card twice is not a spread. The
    client is told which cards came up and where; it is never told the order of
    the rest of the deck, so nothing about the next draw leaks.
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

    return jsonify({
        'ok': True,
        'spread': [
            {'position': POSITIONS[i], 'card': picked[i]}
            for i in range(len(POSITIONS))
        ],
    })
