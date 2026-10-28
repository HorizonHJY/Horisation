#!/usr/bin/env python3.11
"""Print the full prompt the model would receive for a given birth.

Usage:
    python3.11 scripts/bazi_dryrun.py \\
        --date 1998-03-10 --time 21:30 --longitude 86.0 --gender male \\
        --place 石河子 [--today 2026-10-28]

Does not call the vendor. The point is to inspect the fact table and the
rule-corpus hits that go into the user message, so Horizon can tell at a
glance whether the analysis is reasonable before paying for a call.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from Backend.Service.bazi.chart import build_chart              # noqa: E402
from Backend.Service.bazi.analysis import analyze               # noqa: E402
from Backend.Service.bazi import corpus                         # noqa: E402
from Backend.Service.ai.prompts import bazi as bp               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--time', required=True, help='HH:MM (24h, local clock)')
    ap.add_argument('--longitude', type=float, required=True)
    ap.add_argument('--utc-offset', type=float, default=8.0)
    ap.add_argument('--gender', choices=('male', 'female'), required=True)
    ap.add_argument('--place', required=True)
    ap.add_argument('--today', default=None, help='YYYY-MM-DD; defaults to today')
    args = ap.parse_args()

    hh, mm = (int(x) for x in args.time.split(':'))
    birth = datetime.strptime(args.date, '%Y-%m-%d').replace(hour=hh, minute=mm)
    today = date.fromisoformat(args.today) if args.today else date.today()

    chart = build_chart(birth=birth, longitude=args.longitude, utc_offset=args.utc_offset,
                        male=(args.gender == 'male'), place=args.place, today=today)
    a = analyze(chart)

    print('=== Chart ===')
    print(' ' + ' '.join(p['gan'] + p['zhi'] for p in chart['pillars']))
    print(f" 日主 {chart['day_master']['gan']}{chart['day_master']['element']} ·"
          f" 月支 {chart['pillars'][1]['zhi']} · 真太阳时 {chart['solar_time']['true']}")

    print()
    print('=== Fact table ===')
    print(f" 强弱：{a['strength']['level']} (生扶 {a['strength']['friendly_score']} vs"
          f" 克泄耗 {a['strength']['hostile_score']})")
    for r in a['strength']['reasons']:
        print(f"   - {r}")
    print(f" 五行百分比：{a['elements_percent']}")
    print(f" 十神排序：{[(t['star'], t['score']) for t in a['ten_gods']]}")
    print(f" 格局候选：{a['pattern']['name']} — {a['pattern']['basis']}")
    print(f" 用神候选：use={a['gods']['use']} like={a['gods']['like']}"
          f" avoid={a['gods']['avoid']} idle={a['gods']['idle']}")
    for key, items in a['relations']['branch'].items():
        if items:
            print(f" {key}:")
            for r in items:
                print(f"   - {r['detail']} ({r.get('pos') or r.get('at')})"
                      + ('  [on day]' if r.get('on_day') else ''))

    print()
    print('=== Corpus hits ===')
    for h in corpus.hits_for(a):
        print(f" - {h['headline']}: {h['body']}")

    print()
    print('=== Final prompt (user content) ===')
    msgs = bp.build(chart)
    print(msgs[1]['content'])
    print()
    print(f"[system: {len(msgs[0]['content'])} chars; version {bp.PROMPT_VERSION}]")


if __name__ == '__main__':
    main()
