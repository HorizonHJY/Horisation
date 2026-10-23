"""Turn reader feedback into a short correction block for the next reading.

This is the hook. It is deliberately narrow:

  - it reads aggregated reason chips only (never the free-text note, which can
    name someone else's family, job or illness);
  - it says what to avoid, not what to say;
  - it disappears when there is nothing to correct, so nothing is paid for
    on an empty table.

Called from the prompt builders, which is where the model is told how to write.
"""
from __future__ import annotations

from Backend.Controller import feedback_db

# Below this many down votes the tallies are noise — three people marking
# "too vague" is not a trend, and acting on it would overfit to a sample of
# three. The reading still gets its own feedback; it just does not steer yet.
MIN_SAMPLE = 5
MIN_TALLY = 3

_LEAD = {
    'tarot': '前几次的塔罗解读被读者标了「不准」。',
    'bazi': '前几次的八字解读被读者标了「不准」。',
}

_FOLLOW = {
    'tarot': '这次请避开这些问题：把每张牌落到具体的一件事、一个关系、一个正在做的决定上，不要停在牌义层面。',
    'bazi': '这次请避开这些问题：每条判断都点名具体的干支、十神、刑冲合害，并给出可验证的年份，不要停在笼统的性格描述。',
}


def lessons(feature: str) -> str:
    """The block appended to a system prompt, or '' when there is nothing to say."""
    if feature not in feedback_db.FEATURES:
        return ''
    try:
        rows = feedback_db.recent_negative(feature)
    except Exception:
        # Feedback is an improvement, never a reason a reading fails.
        return ''
    if len(rows) < MIN_SAMPLE:
        return ''
    tally: dict[str, int] = {}
    for row in rows:
        for reason in row['reasons']:
            tally[reason] = tally.get(reason, 0) + 1
    hits = [(reason, n) for reason, n in tally.items() if n >= MIN_TALLY]
    if not hits:
        return ''
    hits.sort(key=lambda kv: -kv[1])
    listed = '、'.join(f'「{reason}」{n} 次' for reason, n in hits[:4])
    return (f'\n\n【过往读者的反馈，用于校正，不要复述】\n'
            f'{_LEAD[feature]}抱怨集中在：{listed}。\n'
            f'{_FOLLOW[feature]}')

