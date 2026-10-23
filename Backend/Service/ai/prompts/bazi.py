"""The BaZi reading prompt.

The persona and the four asks are Horizon's own (2026-10-12): a classical
reader working from 子平真诠 / 三命通会 / 滴天髓, ten gods and 生克制化 only,
no psychology and no comfort. Two things are ours:

  - the chart arrives finished (Backend/Service/bazi/chart.py) and the model is
    told not to recompute it — models miscount stems and branches;
  - the answer is JSON in a fixed shape, because the page lays it out as a
    seal, element slips, a luck-pillar track and a year timeline, not prose.
"""

from __future__ import annotations

import json

from ..feedback import lessons
from ..scope import prompt_block

PROMPT_VERSION = 'bazi-v1'

_SYSTEM = """你是一位精通传统四柱八字命理的顶级命理师，熟悉《子平真诠》《三命通会》《滴天髓》等经典体系，只使用八字命理、十神、生克制化、大运流年进行判断，不使用心理学、安慰性语言或模糊占卜话术。

硬规则：
- 排盘已经由程序精确算出（含真太阳时校正），以用户消息里给的为准。不要自行重排，不要改动任何干支、十神、起运时间。
- 判断要有命理依据：每个结论都落到具体的干支、十神、刑冲合害上，例如"卯戌合火""辰戌冲"。
- 只推演已经过去的流年（不晚于"今天"所在年份）；对未来只谈大运的方向，不断言具体事件。
- 回溯事件要写成可以验证的具体陈述（哪一年、哪一类事），不要写成套话。
- 不预测生死、重病、官司输赢、具体投资结果。
- 语言：中文。

输出：只输出一个 JSON 对象，不要任何其他文字，键固定如下——
{
  "verdict": "四到十个字的判词，概括全局，如「印旺身强，食神吐秀」",
  "strength": {
    "level": "身强 / 身弱 / 中和 / 从强 / 从弱，五选一",
    "reasons": ["两到四条依据，每条一句，点名干支"]
  },
  "pattern": {"name": "格局名，如 正印格", "explain": "两到三句：为何成格、成败关键"},
  "gods": {
    "use":   {"element": "用神五行，如 金", "why": "一句"},
    "like":  {"element": "喜神五行", "why": "一句"},
    "avoid": {"element": "忌神五行，可多个用顿号分隔", "why": "一句"},
    "idle":  {"element": "闲神五行，可为空字符串", "why": "一句，可为空字符串"}
  },
  "da_yun": [
    {"ganzhi": "与给定大运逐一对应，只写前六步", "theme": "四个字的主题", "reading": "两到三句：十神、与原局的刑冲合害、吉凶倾向"}
  ],
  "liu_nian": [
    {"year": 2016, "ganzhi": "丙申", "tag": "学业 / 家庭 / 情绪 / 关系 / 事业 / 迁移 / 健康 之一", "event": "一到两句：这一年多半发生了什么，以及命理依据"}
  ],
  "verify": ["四到六条可以用"是/否"回答的验证问题，带年份"],
  "summary": "三到五句总评：一生的主线、最该把握的十年、最该防的一点"
}
liu_nian 挑六到十个最关键的已过流年，按年份从早到晚排列。"""


def _pillar_line(p: dict) -> str:
    hidden = '、'.join(f"{h['gan']}{h['element']}({h['star']})" for h in p['hidden'])
    extra = f"，神煞：{'、'.join(p['shen_sha'])}" if p['shen_sha'] else ''
    return (f"{p['label']}：{p['gan']}{p['zhi']}（天干十神 {p['main_star']}；藏干 {hidden}；"
            f"星运 {p['di_shi']}，自坐 {p['zi_zuo']}；空亡 {p['xun_kong']}；纳音 {p['na_yin']}{extra}）")


def describe(chart: dict) -> str:
    """The chart as the finished table the model reads."""
    y = chart['yun']
    xs = y['xiao_yun']
    da = y['da_yun']
    first = da[0]
    rest = '；'.join(f"{d['start_year']}年{d['ganzhi']}" for d in da[1:])
    current = next((d for d in da if d['current']), None)
    lines = [
        f"①八字信息：{chart['gender_label']}，八字：{' '.join(p['gan'] + p['zhi'] for p in chart['pillars'])}。"
        f"{xs['from_age']}-{xs['to_age']}岁走小运，{first['start_year']}年{first['start_age']}岁起运后第一个大运是{first['ganzhi']}；{rest}。",
        f"起运：出生后 {y['start']['years']} 年 {y['start']['months']} 个月 {y['start']['days']} 天，{y['start']['date']} 交运。",
        f"出生：{chart['solar_time']['clock']}（{chart['input']['place']}），真太阳时 {chart['solar_time']['true']}；农历 {chart['lunar']}。",
        f"日主：{chart['day_master']['gan']}{chart['day_master']['element']}。",
        '',
        '四柱明细：',
        *[_pillar_line(p) for p in chart['pillars']],
        '',
        f"五行加权（天干 1，地支藏干 1/0.5/0.3）：{'，'.join(f'{k}{v}' for k, v in chart['elements'].items())}",
        f"今天：{chart['today']}。" + (f"当前大运：{current['ganzhi']}（{current['start_year']}–{current['end_year']}）。" if current else ''),
        '已过流年（周岁）：' + '；'.join(f"{l['year']}{l['ganzhi']}({l['age']})" for l in chart['liu_nian']),
    ]
    return '\n'.join(lines)


def build(chart: dict) -> list[dict]:
    user = (describe(chart) + "\n\n请完成：② 判断日主强弱、格局、用神、忌神；③ 排出大运与已走过的重要流年；"
            "④ 回溯并推演过去真实发生过的重要事件（学业、家庭、情绪转折、关系、事业节点），用于验证命盘。按规定的 JSON 格式输出。")
    return [{'role': 'system', 'content': _SYSTEM + lessons('bazi')},
            {'role': 'user', 'content': user}]


def _s(v) -> str:
    return v.strip() if isinstance(v, str) else ''


def parse(text: str) -> dict | None:
    """The reply as the structure the page renders, or None if it broke the
    schema — the raw text is still kept and shown."""
    raw = text.strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end <= start:
        return None
    try:
        d = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    try:
        gods = {k: {'element': _s(d['gods'][k].get('element')), 'why': _s(d['gods'][k].get('why'))}
                for k in ('use', 'like', 'avoid', 'idle') if isinstance(d['gods'].get(k), dict)}
        out = {
            'verdict': _s(d['verdict']),
            'strength': {'level': _s(d['strength']['level']),
                         'reasons': [_s(r) for r in d['strength'].get('reasons', []) if _s(r)]},
            'pattern': {'name': _s(d['pattern']['name']), 'explain': _s(d['pattern'].get('explain'))},
            'gods': gods,
            'da_yun': [{'ganzhi': _s(x.get('ganzhi')), 'theme': _s(x.get('theme')), 'reading': _s(x.get('reading'))}
                       for x in d.get('da_yun', []) if isinstance(x, dict)],
            'liu_nian': [{'year': int(x['year']), 'ganzhi': _s(x.get('ganzhi')), 'tag': _s(x.get('tag')),
                          'event': _s(x.get('event'))}
                         for x in d.get('liu_nian', []) if isinstance(x, dict) and str(x.get('year', '')).isdigit()],
            'verify': [_s(v) for v in d.get('verify', []) if _s(v)],
            'summary': _s(d['summary']),
        }
    except (KeyError, TypeError, AttributeError, ValueError):
        return None
    if not (out['verdict'] and out['strength']['level'] and out['pattern']['name'] and 'use' in gods and out['summary']):
        return None
    out['liu_nian'].sort(key=lambda x: x['year'])
    return out


# ── Follow-up questions ──────────────────────────────────────────────
# Asked about one chart, after its reading. The model sees the chart, the
# reading it gave, which of its checks the reader marked right or wrong, and
# the conversation so far — the marks matter most: they are the only ground
# truth about this person the model will ever get.

FOLLOWUP_VERSION = 'bazi-ask-v2'   # v2: scope rules from scope.md (2026-10-12)
MAX_QUESTION_CHARS = 300
HISTORY_TURNS = 8

_FOLLOWUP_SYSTEM = _SYSTEM.split('输出：')[0] + """你已经为这张命盘做过一次完整解读（见下）。现在对方就自己的命盘继续提问。

回答规则：
- 三到六句，直接回答问题，先给结论，再给命理依据（点名具体干支、大运、流年、十神、刑冲合害）。
- 对方标记为「准」的验证项是已知事实，可以作为依据；标记为「不准」的，说明原判断有偏差，回答时要相应修正，不要坚持原说法。
- 问未来：只谈倾向和时间窗口（哪一年、哪步运更有利或更需谨慎），不断言具体事件会发生。
- 问与命盘无关的事（技术问题、闲聊等），一句话说明你只回答与这张命盘相关的问题。
- 不预测生死、重病、官司输赢、具体投资结果；碰到这类问题，回到命盘上该注意的方面。
- 只输出回答本身，纯文本，不要 JSON，不要标题。"""


def _reading_brief(reading: dict | None, raw: str | None) -> str:
    if not reading:
        return (raw or '')[:3000]
    gods = '；'.join(f"{k}：{v['element']}" for k, v in reading['gods'].items() if v.get('element'))
    years = '；'.join(f"{y['year']}{y['ganzhi']} {y['event']}" for y in reading['liu_nian'])
    runs = '；'.join(f"{d['ganzhi']} {d['theme']}：{d['reading']}" for d in reading['da_yun'])
    return '\n'.join([
        f"判词：{reading['verdict']}（{reading['strength']['level']}，{reading['pattern']['name']}）",
        f"用喜忌闲：{gods}",
        f"大运：{runs}",
        f"流年回溯：{years}",
        f"总评：{reading['summary']}",
    ])


def build_followup(payload: dict) -> list[dict]:
    """payload: chart, reading (parsed or None), reading_text, verify
    ({index: 'yes'|'no'}), history [(question, answer)], question."""
    reading = payload.get('reading')
    marks = payload.get('verify') or {}
    checks = reading['verify'] if reading else []
    verified = '\n'.join(
        f"- {q} → {'准' if marks.get(str(i)) == 'yes' else '不准'}"
        for i, q in enumerate(checks) if marks.get(str(i)) in ('yes', 'no')
    ) or '（对方还没有标记）'
    context = (f"{describe(payload['chart'])}\n\n你之前的解读：\n"
               f"{_reading_brief(reading, payload.get('reading_text'))}\n\n对方对验证项的标记：\n{verified}")
    messages = [{'role': 'system', 'content': _FOLLOWUP_SYSTEM + prompt_block()},
                {'role': 'user', 'content': context + '\n\n（以上是背景。接下来是对方的提问。）'},
                {'role': 'assistant', 'content': '明白，请问。'}]
    for q, a in (payload.get('history') or [])[-HISTORY_TURNS:]:
        messages += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
    messages.append({'role': 'user', 'content': payload['question']})
    return messages


def parse_followup(text: str) -> dict | None:
    t = (text or '').strip()
    return {'answer': t} if t else None
