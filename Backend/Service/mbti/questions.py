"""MBTI question banks + scoring.

Two formats:
  - 'short' (20 questions, 5 per dimension) — the 3-minute version
  - 'full'  (60 questions, 15 per dimension) — the 10-minute version

Each question is one statement with two choices. Each choice casts a vote for
one pole of its dimension (E/I, S/N, T/F, J/P). The majority on each pair
decides the letter; ties fall to the first pole in each pair (E, S, T, J),
matching the convention most online tests use.

Written in-house to avoid licence trouble around Myers-Briggs trademark /
paid question banks. The statements are generic enough to pass the smell
test; this is a casual-use tool, not a certified instrument. The page says so.
"""

from __future__ import annotations

DIMENSIONS = [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]
POLES = {p for pair in DIMENSIONS for p in pair}

# ── 20-question short form ─────────────────────────────────────────
# 5 questions per dimension, A/B each voting for one pole.
_SHORT = [
    # Energy — E / I
    ('在陌生场合,我更倾向于', [('主动跟人搭话', 'E'), ('先观察,等别人来找我', 'I')]),
    ('一整天开会后,我需要', [('出去跟人续杯继续聊', 'E'), ('一个人回家关门静一下', 'I')]),
    ('想问题的时候,我喜欢', [('说出来一边聊一边理清', 'E'), ('默默在脑子里想完再开口', 'I')]),
    ('我的朋友圈大多是', [('很多泛泛之交 + 几个深交', 'E'), ('少数几个深交就够了', 'I')]),
    ('接到一个任务时,我会', [('立刻找人讨论', 'E'), ('先自己琢磨方案', 'I')]),

    # Information — S / N
    ('读一本新书我更关注', [('具体的事实和细节', 'S'), ('背后的含义和联系', 'N')]),
    ('做决定时我更相信', [('实际经验和已验证的事', 'S'), ('直觉和潜在可能性', 'N')]),
    ('我说话时常用的词是', [('"具体来说"、"实际上"', 'S'), ('"也许"、"想象一下"', 'N')]),
    ('我更欣赏的朋友是', [('靠谱、务实的那种', 'S'), ('有想法、脑洞大的那种', 'N')]),
    ('面对新技能,我倾向', [('一步步按教程来', 'S'), ('先抓住整体思路再上手', 'N')]),

    # Decisions — T / F
    ('做选择时我更看重', [('逻辑和客观利弊', 'T'), ('对相关人的影响和感受', 'F')]),
    ('朋友跟我倾诉时,我更会', [('帮她分析该怎么办', 'T'), ('先共情,听她说完', 'F')]),
    ('批评别人时我更倾向', [('直说问题,不绕弯', 'T'), ('照顾对方情绪委婉地说', 'F')]),
    ('团队冲突时我会', [('对事不对人,就事论事', 'T'), ('关心大家关系是不是伤了', 'F')]),
    ('读到一个争议新闻,我先想到', [('事实是什么,逻辑对不对', 'T'), ('涉及的人现在感受如何', 'F')]),

    # Structure — J / P
    ('出游前我习惯', [('列好行程、订好一切', 'J'), ('留很多弹性,边走边看', 'P')]),
    ('面对 deadline 我倾向', [('提早做好慢慢推进', 'J'), ('最后冲刺一下搞定', 'P')]),
    ('我的书桌 / 房间大多', [('干净有秩序', 'J'), ('有点乱但我知道哪在哪', 'P')]),
    ('计划被打乱时我会', [('感到不太舒服,想重新安排', 'J'), ('没事,顺势调整就好', 'P')]),
    ('做一个长期项目,我更偏向', [('定好里程碑,按节点推进', 'J'), ('灵活调整,保持多个选项', 'P')]),
]

# ── 60-question full form ──────────────────────────────────────────
# Short form + 10 more per dimension, giving 15 per dimension / 60 total.
_FULL = _SHORT + [
    # E / I extras
    ('开派对时我大多', [('留到最后不想走', 'E'), ('待一会就想找借口早退', 'I')]),
    ('工作休息时我更想', [('找同事聊两句', 'E'), ('戴上耳机独处', 'I')]),
    ('我更擅长', [('口头表达,临场发挥', 'E'), ('书面表达,慢慢写清楚', 'I')]),
    ('对我来说能量来源是', [('跟一群人在一起', 'E'), ('独处时自己的空间', 'I')]),
    ('遇到新鲜事我', [('兴奋想立刻参与', 'E'), ('先观望看别人怎么反应', 'I')]),
    ('没有人在身边一天我会', [('憋得难受想找人说话', 'E'), ('觉得特别放松', 'I')]),
    ('开会时我倾向于', [('积极发言分享想法', 'E'), ('记笔记,等被点名再说', 'I')]),
    ('结束一段关系我更倾向', [('说清楚讲明白', 'E'), ('自己想通慢慢淡掉', 'I')]),
    ('面对一个新 group chat', [('立刻参与讨论', 'E'), ('先潜水观察气氛', 'I')]),
    ('我更害怕', [('长时间一个人', 'E'), ('一直被别人围着', 'I')]),

    # S / N extras
    ('我更擅长记住', [('具体的日期、地点、价格', 'S'), ('人物关系和故事脉络', 'N')]),
    ('给我讲故事,我希望听到', [('先后发生了什么细节', 'S'), ('这个故事想说明什么', 'N')]),
    ('我更喜欢的书是', [('纪实、历史、工具类', 'S'), ('科幻、象征、思想类', 'N')]),
    ('面对一个新项目我先', [('看现有方案是怎么做的', 'S'), ('想一个全新的切入角度', 'N')]),
    ('"创意"对我来说更接近', [('对已知的巧妙改良', 'S'), ('完全脱离现实的发散', 'N')]),
    ('我更相信', [('亲眼看到、亲手摸到的', 'S'), ('第一感觉和模式识别', 'N')]),
    ('我说明一件事喜欢用', [('具体的例子和数字', 'S'), ('类比和隐喻', 'N')]),
    ('在工作中我更擅长', [('执行、落地、把事做实', 'S'), ('战略、方向、从 0 到 1', 'N')]),
    ('面对未解的问题我倾向', [('耐心找资料积累事实', 'S'), ('直接猜一个结论再验证', 'N')]),
    ('我对未来的感觉', [('下个月要做什么我知道', 'S'), ('长远我有一个愿景,近期常变', 'N')]),

    # T / F extras
    ('我更欣赏别人的', [('能力和判断力', 'T'), ('善良和体贴', 'F')]),
    ('面对不公平的事我先', [('分析是不是真的不公平', 'T'), ('感到愤怒想为当事人说话', 'F')]),
    ('给朋友建议我倾向', [('直接讲我的真实看法', 'T'), ('先理解对方立场再说', 'F')]),
    ('读一篇观点文章我看', [('论证严不严谨', 'T'), ('有没有站在人的角度', 'F')]),
    ('我更难接受别人', [('不讲道理', 'T'), ('冷漠不照顾感受', 'F')]),
    ('做选择时我希望', [('基于事实不被情绪左右', 'T'), ('重视相关人的感受', 'F')]),
    ('看到朋友哭了我会先', [('问发生了什么,帮她想办法', 'T'), ('陪她坐着,让她哭会', 'F')]),
    ('职场上我更看重', [('能力匹配、晋升合理', 'T'), ('氛围和睦、互相照顾', 'F')]),
    ('我更难理解', [('情绪化的决定', 'T'), ('冷冰冰的理性', 'F')]),
    ('评价别人我倾向', [('基于他做了什么', 'T'), ('基于他是什么样的人', 'F')]),

    # J / P extras
    ('我的日历一般是', [('提前排满,按计划走', 'J'), ('大致有事,具体当天看', 'P')]),
    ('开始新任务前我', [('先列好步骤', 'J'), ('直接动手,边做边想', 'P')]),
    ('我更舒服的是', [('事情有定论,结束了', 'J'), ('事情还开着,保留可能', 'P')]),
    ('面对多个选项我', [('尽早选一个定下来', 'J'), ('想多留一些时间比较', 'P')]),
    ('下班回家我会', [('做个 to-do 规划明天', 'J'), ('随意刷手机看看', 'P')]),
    ('出差前一周我', [('已经收好行李', 'J'), ('出发前一晚才开始收', 'P')]),
    ('有人临时改约会我', [('有点不爽,但接受', 'J'), ('完全无所谓,挺好', 'P')]),
    ('工作时我倾向', [('定好规则严格执行', 'J'), ('摸索着调整规则', 'P')]),
    ('面对未知我更想', [('尽快掌握全貌', 'J'), ('享受探索的过程', 'P')]),
    ('我更喜欢的生活节奏', [('稳定、可预测', 'J'), ('有变化、有惊喜', 'P')]),
]

assert len(_SHORT) == 20
assert len(_FULL)  == 60
# Sanity: 5 per dim short, 15 per dim full.
_dims_short = [c[1] for q in _SHORT for c in q[1]]
_dims_full  = [c[1] for q in _FULL  for c in q[1]]
for pair in DIMENSIONS:
    for p in pair:
        assert _dims_short.count(p) == 5,  f'short imbalance: {p}'
        assert _dims_full.count(p)  == 15, f'full imbalance: {p}'


def _to_dict(bank: list) -> list[dict]:
    """Turn (statement, [(choice, pole), ...]) into JSON-friendly dicts."""
    return [
        {
            'id': i,
            'statement': q[0],
            'choices': [{'label': c, 'pole': p} for c, p in q[1]],
        }
        for i, q in enumerate(bank)
    ]


def get_bank(mode: str) -> list[dict]:
    if mode == 'short':
        return _to_dict(_SHORT)
    if mode == 'full':
        return _to_dict(_FULL)
    raise ValueError(f'unknown mbti mode: {mode}')


def score(mode: str, answers: list[int]) -> dict:
    """`answers` is a list of 0/1 choice indices, one per question in order.
    Returns {type: 'INFP', scores: {E:3,I:17,...}, raw_counts: ...}.
    Tie breaks fall to E / S / T / J (common convention)."""
    bank = _SHORT if mode == 'short' else _FULL
    if len(answers) != len(bank):
        raise ValueError(f'expected {len(bank)} answers, got {len(answers)}')
    counts = {p: 0 for p in POLES}
    for q, a in zip(bank, answers):
        choices = q[1]
        if a not in (0, 1):
            raise ValueError(f'answer must be 0 or 1, got {a}')
        pole = choices[a][1]
        counts[pole] += 1
    letters = []
    for a, b in DIMENSIONS:
        letters.append(a if counts[a] >= counts[b] else b)
    return {
        'type':   ''.join(letters),
        'counts': counts,
        'mode':   mode,
    }
