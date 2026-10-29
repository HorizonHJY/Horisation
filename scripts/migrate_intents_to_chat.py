#!/usr/bin/env python3
"""Migrate trade_intents → chat system bubbles.

Run with --dry-run first. --apply does the write and drops the intents table.

For every intent, insert a system bubble `[sys:interest]<json>` as the buyer's
chat message in the pair's room_key. If the intent had a `message`, append a
second normal buyer message with that text. Terminal states (completed /
cancelled / declined) keep the memory of what happened via a `status` field
in the system bubble payload so the chat history still reads right.
"""
import argparse
import json
import sqlite3
import sys
import uuid


def pair_key(a: str, b: str) -> str:
    return ':'.join(sorted([a, b]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('db', help='Path to market.db')
    ap.add_argument('--apply', action='store_true', help='Actually write (default: dry run)')
    ap.add_argument('--drop-table', action='store_true', help='After apply, drop trade_intents table')
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Snapshot counts for the plan line.
    c.execute('SELECT status, COUNT(*) FROM trade_intents GROUP BY status')
    counts = dict(c.fetchall())
    total = sum(counts.values())
    print(f'intents: total={total} by_status={counts}')

    c.execute("SELECT COUNT(*) FROM listings WHERE status='reserved'")
    reserved_now = c.fetchone()[0]
    print(f'reserved listings today: {reserved_now} (will be flipped back to active)')

    # Pull every intent + its listing title/price so the bubble reads well
    # even if the listing is later edited or deleted.
    c.execute("""
        SELECT i.id, i.listing_id, i.buyer, i.seller, i.status, i.message, i.created_at,
               l.title, l.price
        FROM trade_intents i
        LEFT JOIN listings l ON l.id = i.listing_id
        ORDER BY i.created_at
    """)
    rows = c.fetchall()

    plan = []
    for r in rows:
        room = pair_key(r['buyer'], r['seller'])
        payload = {
            'listing_id': r['listing_id'],
            'title': r['title'] or '(deleted listing)',
            'price': r['price'],
            'status': r['status'],   # pending/accepted/completed/cancelled/declined
        }
        plan.append({
            'room_key': room,
            'sender': r['buyer'],
            'created_at': r['created_at'],
            'system_content': f"[sys:interest]{json.dumps(payload, ensure_ascii=False)}",
            'followup_text': r['message'],
        })

    for p in plan:
        print('  ·', p['room_key'], '←', p['sender'],
              '|', p['system_content'][:120], '| note:', p['followup_text'])

    if not args.apply:
        print('(dry-run — no writes)')
        return 0

    # Insert the bubbles. Preserve `created_at` so the bubbles land in history
    # at the moment the intent was first created.
    writes = 0
    for p in plan:
        c.execute(
            'INSERT INTO private_chat_messages (id, room_key, sender, content, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (str(uuid.uuid4()), p['room_key'], p['sender'], p['system_content'], p['created_at'])
        )
        writes += 1
        if p['followup_text']:
            # The buyer's own words land one second later so the system bubble
            # comes first in the thread.
            c.execute(
                "INSERT INTO private_chat_messages (id, room_key, sender, content, created_at) "
                "VALUES (?, ?, ?, ?, datetime(?, '+1 second'))",
                (str(uuid.uuid4()), p['room_key'], p['sender'], p['followup_text'], p['created_at'])
            )
            writes += 1

    c.execute("UPDATE listings SET status='active' WHERE status='reserved'")
    unreserved = c.rowcount

    if args.drop_table:
        c.execute('DROP TABLE IF EXISTS trade_intents')
        print('dropped trade_intents table')

    conn.commit()
    print(f'applied: wrote {writes} chat rows, unreserved {unreserved} listings')
    return 0


if __name__ == '__main__':
    sys.exit(main())
