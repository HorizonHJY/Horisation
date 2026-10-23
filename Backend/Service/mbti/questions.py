"""MBTI question banks + scoring. Bilingual (Chinese / English).

Two formats:
  - 'short' (20 questions, 5 per dimension) — the 3-minute version
  - 'full'  (60 questions, 15 per dimension) — the 10-minute version

Each question is one statement with two choices. Each choice casts a vote for
one pole of its dimension (E/I, S/N, T/F, J/P). The majority on each pair
decides the letter; ties fall to the first pole in each pair (E, S, T, J),
matching the convention most online tests use.

Both languages ride the same tuple so the page can switch without a second
fetch: (statement_zh, statement_en, [(choice_zh, choice_en, pole), ...]).
The English is our own translation, written to read naturally for a native
speaker rather than mirror the Chinese word for word.

Written in-house to avoid licence trouble around Myers-Briggs trademark /
paid question banks. This is a casual-use tool, not a certified instrument;
the page says so.
"""

from __future__ import annotations

DIMENSIONS = [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]
POLES = {p for pair in DIMENSIONS for p in pair}

# ── 20-question short form ─────────────────────────────────────────
# 5 questions per dimension, A/B each voting for one pole.
_SHORT = [
    # Energy — E / I
    ('在陌生场合,我更倾向于', "In unfamiliar settings, I tend to", [
        ('主动跟人搭话', "start conversations with people", 'E'),
        ('先观察,等别人来找我', "observe and wait for people to come to me", 'I')]),
    ('一整天开会后,我需要', "After a full day of meetings, I need to", [
        ('出去跟人续杯继续聊', "go out and keep talking over drinks", 'E'),
        ('一个人回家关门静一下', "go home, shut the door, and be alone for a bit", 'I')]),
    ('想问题的时候,我喜欢', "When thinking through a problem, I prefer to", [
        ('说出来一边聊一边理清', "talk it out as I go", 'E'),
        ('默默在脑子里想完再开口', "think it through in my head before speaking", 'I')]),
    ('我的朋友圈大多是', "My circle of friends is mostly", [
        ('很多泛泛之交 + 几个深交', "many acquaintances plus a few close friends", 'E'),
        ('少数几个深交就够了', "a few close friends is enough", 'I')]),
    ('接到一个任务时,我会', "When I get a task, I", [
        ('立刻找人讨论', "immediately discuss it with others", 'E'),
        ('先自己琢磨方案', "mull over a plan on my own first", 'I')]),

    # Information — S / N
    ('读一本新书我更关注', "In a new book I pay more attention to", [
        ('具体的事实和细节', "concrete facts and details", 'S'),
        ('背后的含义和联系', "the meaning and connections behind it", 'N')]),
    ('做决定时我更相信', "When deciding, I trust more", [
        ('实际经验和已验证的事', "real experience and what is already proven", 'S'),
        ('直觉和潜在可能性', "intuition and what could be", 'N')]),
    ('我说话时常用的词是', "The words I lean on are", [
        ('"具体来说"、"实际上"', '"to be specific", "actually"', 'S'),
        ('"也许"、"想象一下"', '"maybe", "imagine if"', 'N')]),
    ('我更欣赏的朋友是', "The friends I admire are", [
        ('靠谱、务实的那种', "the reliable, practical kind", 'S'),
        ('有想法、脑洞大的那种', "the imaginative, big-idea kind", 'N')]),
    ('面对新技能,我倾向', "With a new skill, I tend to", [
        ('一步步按教程来', "follow the tutorial step by step", 'S'),
        ('先抓住整体思路再上手', "grasp the big picture before diving in", 'N')]),

    # Decisions — T / F
    ('做选择时我更看重', "When choosing, I weigh more", [
        ('逻辑和客观利弊', "logic and objective trade-offs", 'T'),
        ('对相关人的影响和感受', "how it affects people and their feelings", 'F')]),
    ('朋友跟我倾诉时,我更会', "When a friend vents to me, I", [
        ('帮她分析该怎么办', "help them figure out what to do", 'T'),
        ('先共情,听她说完', "empathize first and just listen", 'F')]),
    ('批评别人时我更倾向', "When giving criticism, I", [
        ('直说问题,不绕弯', "say it straight, no beating around the bush", 'T'),
        ('照顾对方情绪委婉地说', "soften it to protect their feelings", 'F')]),
    ('团队冲突时我会', "In a team conflict, I", [
        ('对事不对人,就事论事', "focus on the issue, not the person", 'T'),
        ('关心大家关系是不是伤了', "worry about whether relationships are hurt", 'F')]),
    ('读到一个争议新闻,我先想到', "Reading a controversial story, I first think of", [
        ('事实是什么,逻辑对不对', "the facts and whether the logic holds", 'T'),
        ('涉及的人现在感受如何', "how the people involved feel right now", 'F')]),

    # Structure — J / P
    ('出游前我习惯', "Before a trip, I usually", [
        ('列好行程、订好一切', "plan the itinerary and book everything", 'J'),
        ('留很多弹性,边走边看', "leave room and figure it out as I go", 'P')]),
    ('面对 deadline 我倾向', "Facing a deadline, I", [
        ('提早做好慢慢推进', "start early and pace myself", 'J'),
        ('最后冲刺一下搞定', "sprint at the end to get it done", 'P')]),
    ('我的书桌 / 房间大多', "My desk / room is mostly", [
        ('干净有秩序', "clean and orderly", 'J'),
        ('有点乱但我知道哪在哪', "a bit messy but I know where things are", 'P')]),
    ('计划被打乱时我会', "When plans get disrupted, I", [
        ('感到不太舒服,想重新安排', "feel uneasy and want to reschedule", 'J'),
        ('没事,顺势调整就好', "am fine with it and adjust", 'P')]),
    ('做一个长期项目,我更偏向', "On a long project, I prefer to", [
        ('定好里程碑,按节点推进', "set milestones and move by them", 'J'),
        ('灵活调整,保持多个选项', "stay flexible and keep options open", 'P')]),
]

# ── 60-question full form ──────────────────────────────────────────
# Short form + 10 more per dimension, giving 15 per dimension / 60 total.
_FULL = _SHORT + [
    # E / I extras
    ('开派对时我大多', "At a party I usually", [
        ('留到最后不想走', "stay until the end, not wanting to leave", 'E'),
        ('待一会就想找借口早退', "want to slip out early after a while", 'I')]),
    ('工作休息时我更想', "On a work break, I would rather", [
        ('找同事聊两句', "chat with a coworker", 'E'),
        ('戴上耳机独处', "put on headphones and be alone", 'I')]),
    ('我更擅长', "I am better at", [
        ('口头表达,临场发挥', "speaking on the spot", 'E'),
        ('书面表达,慢慢写清楚', "writing things out clearly", 'I')]),
    ('对我来说能量来源是', "My energy comes from", [
        ('跟一群人在一起', "being around people", 'E'),
        ('独处时自己的空间', "time alone in my own space", 'I')]),
    ('遇到新鲜事我', "When something new happens, I", [
        ('兴奋想立刻参与', "get excited and jump in", 'E'),
        ('先观望看别人怎么反应', "hang back and watch how others react", 'I')]),
    ('没有人在身边一天我会', "A day with no one around leaves me", [
        ('憋得难受想找人说话', "restless, wanting to talk to someone", 'E'),
        ('觉得特别放松', "feeling especially relaxed", 'I')]),
    ('开会时我倾向于', "In meetings I tend to", [
        ('积极发言分享想法', "speak up and share ideas", 'E'),
        ('记笔记,等被点名再说', "take notes and wait to be called on", 'I')]),
    ('结束一段关系我更倾向', "Ending a relationship, I would rather", [
        ('说清楚讲明白', "talk it through and be clear", 'E'),
        ('自己想通慢慢淡掉', "think it through and let it fade", 'I')]),
    ('面对一个新 group chat', "In a new group chat, I", [
        ('立刻参与讨论', "jump into the discussion", 'E'),
        ('先潜水观察气氛', "lurk and read the room first", 'I')]),
    ('我更害怕', "I fear more", [
        ('长时间一个人', "being alone for a long time", 'E'),
        ('一直被别人围着', "being constantly surrounded", 'I')]),

    # S / N extras
    ('我更擅长记住', "I am better at remembering", [
        ('具体的日期、地点、价格', "specific dates, places, prices", 'S'),
        ('人物关系和故事脉络', "relationships and story arcs", 'N')]),
    ('给我讲故事,我希望听到', "In a story, I want to hear", [
        ('先后发生了什么细节', "what happened, in order, in detail", 'S'),
        ('这个故事想说明什么', "what the story is trying to say", 'N')]),
    ('我更喜欢的书是', "I prefer books that are", [
        ('纪实、历史、工具类', "factual, historical, or practical", 'S'),
        ('科幻、象征、思想类', "speculative, symbolic, or philosophical", 'N')]),
    ('面对一个新项目我先', "On a new project, I first", [
        ('看现有方案是怎么做的', "see how it is done now", 'S'),
        ('想一个全新的切入角度', "imagine a fresh angle", 'N')]),
    ('"创意"对我来说更接近', '"Creativity" to me is closer to', [
        ('对已知的巧妙改良', "a clever improvement on what exists", 'S'),
        ('完全脱离现实的发散', "freewheeling far from reality", 'N')]),
    ('我更相信', "I trust more", [
        ('亲眼看到、亲手摸到的', "what I can see and touch", 'S'),
        ('第一感觉和模式识别', "first impressions and pattern recognition", 'N')]),
    ('我说明一件事喜欢用', "To explain something, I use", [
        ('具体的例子和数字', "concrete examples and numbers", 'S'),
        ('类比和隐喻', "analogies and metaphors", 'N')]),
    ('在工作中我更擅长', "At work I am better at", [
        ('执行、落地、把事做实', "executing and getting things done", 'S'),
        ('战略、方向、从 0 到 1', "strategy, direction, zero to one", 'N')]),
    ('面对未解的问题我倾向', "With an unsolved problem, I", [
        ('耐心找资料积累事实', "patiently gather facts", 'S'),
        ('直接猜一个结论再验证', "guess a conclusion then test it", 'N')]),
    ('我对未来的感觉', "About the future, I", [
        ('下个月要做什么我知道', "know what I am doing next month", 'S'),
        ('长远我有一个愿景,近期常变', "have a long-term vision, short-term shifts", 'N')]),

    # T / F extras
    ('我更欣赏别人的', "What I admire in others is", [
        ('能力和判断力', "competence and judgment", 'T'),
        ('善良和体贴', "kindness and consideration", 'F')]),
    ('面对不公平的事我先', "Facing something unfair, I first", [
        ('分析是不是真的不公平', "analyze whether it really is unfair", 'T'),
        ('感到愤怒想为当事人说话', "feel angry and want to speak up", 'F')]),
    ('给朋友建议我倾向', "Giving a friend advice, I", [
        ('直接讲我的真实看法', "say what I really think", 'T'),
        ('先理解对方立场再说', "understand their side first", 'F')]),
    ('读一篇观点文章我看', "In an opinion piece, I look at", [
        ('论证严不严谨', "whether the reasoning is sound", 'T'),
        ('有没有站在人的角度', "whether it sees the human side", 'F')]),
    ('我更难接受别人', "I find it harder to accept when someone is", [
        ('不讲道理', "unreasonable", 'T'),
        ('冷漠不照顾感受', "cold and inconsiderate", 'F')]),
    ('做选择时我希望', "When deciding, I want to", [
        ('基于事实不被情绪左右', "go on facts, not feelings", 'T'),
        ('重视相关人的感受', "weigh people's feelings", 'F')]),
    ('看到朋友哭了我会先', "Seeing a friend cry, I first", [
        ('问发生了什么,帮她想办法', "ask what happened and help solve it", 'T'),
        ('陪她坐着,让她哭会', "sit with her and let her cry", 'F')]),
    ('职场上我更看重', "At work I value more", [
        ('能力匹配、晋升合理', "fair matching of skill and promotion", 'T'),
        ('氛围和睦、互相照顾', "a warm, caring culture", 'F')]),
    ('我更难理解', "I find it harder to understand", [
        ('情绪化的决定', "emotional decisions", 'T'),
        ('冷冰冰的理性', "cold rationality", 'F')]),
    ('评价别人我倾向', "When judging others, I go by", [
        ('基于他做了什么', "what they have done", 'T'),
        ('基于他是什么样的人', "who they are", 'F')]),

    # J / P extras
    ('我的日历一般是', "My calendar is usually", [
        ('提前排满,按计划走', "booked in advance, run by plan", 'J'),
        ('大致有事,具体当天看', "loose, filled in day by day", 'P')]),
    ('开始新任务前我', "Before a new task, I", [
        ('先列好步骤', "list the steps first", 'J'),
        ('直接动手,边做边想', "dive in and figure it out as I go", 'P')]),
    ('我更舒服的是', "I am more comfortable when", [
        ('事情有定论,结束了', "things are settled and done", 'J'),
        ('事情还开着,保留可能', "things are still open with options", 'P')]),
    ('面对多个选项我', "With several options, I", [
        ('尽早选一个定下来', "pick one early and commit", 'J'),
        ('想多留一些时间比较', "keep comparing a bit longer", 'P')]),
    ('下班回家我会', "After work, I", [
        ('做个 to-do 规划明天', "make a to-do list for tomorrow", 'J'),
        ('随意刷手机看看', "scroll my phone idly", 'P')]),
    ('出差前一周我', "A week before a trip, I have", [
        ('已经收好行李', "already packed", 'J'),
        ('出发前一晚才开始收', "not packed until the night before", 'P')]),
    ('有人临时改约会我', "If someone reschedules last-minute, I", [
        ('有点不爽,但接受', "am a bit annoyed but accept it", 'J'),
        ('完全无所谓,挺好', "do not mind at all", 'P')]),
    ('工作时我倾向', "At work I prefer to", [
        ('定好规则严格执行', "set rules and follow them strictly", 'J'),
        ('摸索着调整规则', "figure rules out as I go", 'P')]),
    ('面对未知我更想', "Facing the unknown, I want to", [
        ('尽快掌握全貌', "grasp the full picture soon", 'J'),
        ('享受探索的过程', "enjoy the exploring", 'P')]),
    ('我更喜欢的生活节奏', "I prefer a pace of life that is", [
        ('稳定、可预测', "steady and predictable", 'J'),
        ('有变化、有惊喜', "varied with surprises", 'P')]),
]

assert len(_SHORT) == 20
assert len(_FULL)  == 60
# Sanity: 5 per dim short, 15 per dim full.
_dims_short = [c[2] for q in _SHORT for c in q[2]]
_dims_full  = [c[2] for q in _FULL  for c in q[2]]
for pair in DIMENSIONS:
    for p in pair:
        assert _dims_short.count(p) == 5,  f'short imbalance: {p}'
        assert _dims_full.count(p)  == 15, f'full imbalance: {p}'


def _to_dict(bank: list) -> list[dict]:
    """Turn the bilingual tuples into JSON-friendly dicts."""
    return [
        {
            'id': i,
            'statement': q[0],
            'statement_en': q[1],
            'choices': [{'label': c[0], 'label_en': c[1], 'pole': c[2]} for c in q[2]],
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
    Returns {type: 'INFP', counts: {E:3,I:17,...}, mode: ...}.
    Tie breaks fall to E / S / T / J (common convention)."""
    bank = _SHORT if mode == 'short' else _FULL
    if len(answers) != len(bank):
        raise ValueError(f'expected {len(bank)} answers, got {len(answers)}')
    counts = {p: 0 for p in POLES}
    for q, a in zip(bank, answers):
        choices = q[2]
        if a not in (0, 1):
            raise ValueError(f'answer must be 0 or 1, got {a}')
        pole = choices[a][2]
        counts[pole] += 1
    letters = []
    for a, b in DIMENSIONS:
        letters.append(a if counts[a] >= counts[b] else b)
    return {
        'type':   ''.join(letters),
        'counts': counts,
        'mode':   mode,
    }

