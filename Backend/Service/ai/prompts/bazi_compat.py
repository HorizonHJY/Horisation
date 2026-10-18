"""The BaZi 合盘 reading prompt.

Same persona as 八字 — a classical reader working from 子平真诠 / 三命通会 /
滴天髓 — but the subject is two charts, not one. The arithmetic (日柱合冲、
生肖关系、五行互补、十神) is already done by Backend/Service/bazi/compat.py;
the model reads two finished tables and the pre-computed relations, and gives
the reading.
"""

from __future__ import annotations

import json

from ..scope import prompt_block
from . import bazi as bazi_prompt   # reuse the describe() for a single chart

PROMPT_VERSION = 'bazi-compat-v1'

_SYSTEM = """你是一位精通传统四柱八字命理的顶级命理师，熟悉《子平真诠》《三命通会》《滴天髓》等经典体系。你擅长"合盘"——同时看两张命盘，判断两人之间的缘分深浅、相处方式、互补与冲克之处，以及要注意的时间节点。

硬规则：
- 两张命盘和它们之间的关系（日柱天干地支的合/冲/克、生肖、五行互补、十神）已由程序精确算出，用户消息里给的为准。不要自行重排。
- 判断要有命理依据：每条结论都落到具体的干支、十神、合冲上，例如"日柱甲己合土，天合地合为上婚"。
- 不断言婚姻必成、必离、生死、重病；碰到此类问题，回到命盘上该注意的部分。
- 同性合盘按友情/合作盘看待，不要擅自判定关系性质。
- 语言：中文。

输出：只输出一个 JSON 对象，不要任何其他文字，键固定如下——
{
  "verdict": "八到十二字的判词，概括两人缘分，如「天合地合，补益为重」",
  "level": "上等 / 中上 / 中等 / 中下 / 下等 五选一",
  "chemistry": {
    "summary": "两到三句：这段关系的核心性质——是互补、是磨合、是激发、还是消耗",
    "reasons": ["两到四条依据，每条一句，点名干支或十神"]
  },
  "day_pillar": "两到三句：日柱相合/相冲/相克说明了什么，对夫妻宫/感情位的影响",
  "complement": "两到三句：五行互补的方向——谁补谁，缺什么、旺什么",
  "ten_god": "两到三句：对方日干对本命的十神关系意味着什么（如正官 → 规范我、七杀 → 压制我、正财 → 滋养我）",
  "zodiac": "一两句：生肖合冲的加成或扣分",
  "strengths": ["三到五条：这段关系最有力量的方面"],
  "cautions":  ["三到五条：要注意的地方——不是预言不幸，而是提醒"],
  "advice": "三到五句：相处建议、如何扬长避短；给具体可行的、落到行为层面的建议",
  "verify": ["四到六条可以用"是/否"回答的验证问题，关于两人实际相处中的现象，带时间或情境"],
  "summary": "三到四句总评：这段关系的主线、最该把握的、最该防的"
}"""


def _one(chart: dict, label: str) -> str:
    """一张命盘的描述，带标签「本命 / 对象」。"""
    body = bazi_prompt.describe(chart)
    return f"=== {label} ===\n" + body


def _rel_lines(compat: dict) -> str:
    d = compat['day_pair']
    rels = []
    for r in d['gan_rel']:
        rels.append(f"日干关系：{r['detail']}")
    for r in d['zhi_rel']:
        rels.append(f"日支关系：{r['detail']}")
    for r in compat['zodiac']['rel']:
        rels.append(f"生肖关系：{r['detail']}")
    el = compat['elements']
    if el['gift_to_self']:
        rels.append("对方对本命的五行补益：" + '、'.join(f"{k}+{v}" for k, v in el['gift_to_self'].items()))
    if el['gift_to_other']:
        rels.append("本命对对方的五行补益：" + '、'.join(f"{k}+{v}" for k, v in el['gift_to_other'].items()))
    tg = compat['ten_gods']
    rels.append(f"对方日干 {compat['day_pair']['other']['gan']} 对本命（{tg['self_me_label']}）的十神：{tg['spouse_god']}")
    cells = '、'.join(f"{c['pillar']}{c['gan']}→{c['ten_god']}" for c in tg['cells'])
    rels.append(f"对方四柱天干对本命日主的十神：{cells}")
    rels.append(f"配对类型：{'异性/情缘盘' if compat['gender_pair'] == 'romantic' else '同性/友情合作盘'}")
    return '\n'.join(f"- {r}" for r in rels)


def build(payload: dict) -> list[dict]:
    """payload: chart_a (本命), chart_b (对象), compat (compat.build_compat 输出)"""
    body = '\n\n'.join([
        _one(payload['chart_a'], '本命（左）'),
        _one(payload['chart_b'], '对象（右）'),
        '=== 已算好的合盘关系 ===',
        _rel_lines(payload['compat']),
    ])
    user = body + (
        "\n\n请完成合盘解读：说明两人缘分性质、日柱关系、五行互补、十神关系、"
        "生肖加减分、各自的优势与要注意的地方，并给出可行的相处建议。按规定的 JSON 格式输出。"
    )
    return [{'role': 'system', 'content': _SYSTEM}, {'role': 'user', 'content': user}]


def _s(v) -> str:
    return v.strip() if isinstance(v, str) else ''


def parse(text: str) -> dict | None:
    raw = text.strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end <= start: return None
    try:
        d = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    if not isinstance(d, dict): return None
    try:
        chem = d.get('chemistry', {}) or {}
        out = {
            'verdict': _s(d.get('verdict')),
            'level':   _s(d.get('level')),
            'chemistry': {
                'summary': _s(chem.get('summary')),
                'reasons': [_s(r) for r in chem.get('reasons', []) if _s(r)],
            },
            'day_pillar': _s(d.get('day_pillar')),
            'complement': _s(d.get('complement')),
            'ten_god':    _s(d.get('ten_god')),
            'zodiac':     _s(d.get('zodiac')),
            'strengths':  [_s(x) for x in d.get('strengths', []) if _s(x)],
            'cautions':   [_s(x) for x in d.get('cautions', []) if _s(x)],
            'advice':     _s(d.get('advice')),
            'verify':     [_s(x) for x in d.get('verify', []) if _s(x)],
            'summary':    _s(d.get('summary')),
        }
    except (KeyError, TypeError, AttributeError, ValueError):
        return None
    if not (out['verdict'] and out['level'] and out['summary'] and out['chemistry']['summary']):
        return None
    return out


# ── Follow-up questions on a 合盘 ─────────────────────────────────

FOLLOWUP_VERSION = 'bazi-compat-ask-v1'
MAX_QUESTION_CHARS = 300
HISTORY_TURNS = 8

_FOLLOWUP_SYSTEM = _SYSTEM.split('输出：')[0] + """你已经为这对合盘做过一次完整解读（见下）。现在用户就这段关系继续提问。

回答规则：
- 三到六句，直接回答，先给结论，再给命理依据（点名干支、合冲、十神、五行）。
- 用户标记为「准」的验证项是已知事实，可作为依据；标记为「不准」的，说明原判断有偏差，回答时相应修正。
- 问未来：只谈倾向和时间窗口（哪年、哪步大运更有利或需谨慎），不断言具体结果。
- 问与合盘无关的事，一句话说明只回答这段关系相关的问题。
- 不预测生死、重病、婚姻必成/必离、具体投资结果。
- 只输出回答本身，纯文本，不要 JSON、不要标题。"""


def _reading_brief(reading: dict | None, raw: str | None) -> str:
    if not reading:
        return (raw or '')[:3000]
    return '\n'.join([
        f"判词：{reading['verdict']}（{reading['level']}）",
        f"关系核心：{reading['chemistry']['summary']}",
        f"日柱：{reading['day_pillar']}",
        f"五行互补：{reading['complement']}",
        f"十神：{reading['ten_god']}",
        f"生肖：{reading['zodiac']}",
        "优势：" + '；'.join(reading['strengths']),
        "要注意：" + '；'.join(reading['cautions']),
        f"建议：{reading['advice']}",
        f"总评：{reading['summary']}",
    ])


def build_followup(payload: dict) -> list[dict]:
    """payload: chart_a, chart_b, compat, reading (parsed|None), reading_text,
    verify ({index: 'yes'|'no'}), history [(q,a)], question."""
    reading = payload.get('reading')
    marks = payload.get('verify') or {}
    checks = reading['verify'] if reading else []
    verified = '\n'.join(
        f"- {q} → {'准' if marks.get(str(i)) == 'yes' else '不准'}"
        for i, q in enumerate(checks) if marks.get(str(i)) in ('yes', 'no')
    ) or '（用户还没有标记）'
    body = '\n\n'.join([
        _one(payload['chart_a'], '本命（左）'),
        _one(payload['chart_b'], '对象（右）'),
        '合盘关系：\n' + _rel_lines(payload['compat']),
        '你之前的解读：\n' + _reading_brief(reading, payload.get('reading_text')),
        '用户对验证项的标记：\n' + verified,
    ])
    messages = [{'role': 'system', 'content': _FOLLOWUP_SYSTEM + prompt_block()},
                {'role': 'user', 'content': body + '\n\n（以上是背景。接下来是用户的提问。）'},
                {'role': 'assistant', 'content': '明白，请问。'}]
    for q, a in (payload.get('history') or [])[-HISTORY_TURNS:]:
        messages += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
    messages.append({'role': 'user', 'content': payload['question']})
    return messages


def parse_followup(text: str) -> dict | None:
    t = (text or '').strip()
    return {'answer': t} if t else None
