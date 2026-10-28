"""The BaZi reading prompt.

v2 (2026-10-28) — accuracy pass. Chart arithmetic is still done by
`Backend/Service/bazi/chart.py`. What changed:

  - A fact table is produced by `Backend/Service/bazi/analysis.py` (strength,
    weighted elements, ten-god ranking, 刑冲合害, 格局候选, 用神候选) and
    injected into the user message as 「权威事实表」. The model is told,
    verbatim, not to contradict it: each judgement must point to a干支 /
    十神 / relation present in the table.
  - A small rule corpus (`Backend/Service/bazi/corpus.py`) is matched against
    the fact table and injected as 「可用短语」. These are plain-Chinese
    summaries of common configurations; the model may echo them but must not
    invent sources. Classical book names and citations are forbidden on
    screen (prompt rule).
  - A validator (`validate` below) checks the parsed reply against the fact
    table and returns a list of contradictions. The caller may retry once
    with the contradictions appended; still-wrong items are dropped.

The persona and the four asks are Horizon's own; the four-pillar table is
still handed over finished; the JSON schema is unchanged so the front end
(印鉴 / 五行条 / 大运轨 / 流年时间轴) keeps working.
"""

from __future__ import annotations

import json

from ..feedback import lessons
from ..scope import prompt_block
from ...bazi.analysis import analyze
from ...bazi import corpus

PROMPT_VERSION = 'bazi-v2'

_SYSTEM = """你是一位精通传统四柱八字命理的顶级命理师，只使用八字命理、十神、生克制化、大运流年进行判断，不使用心理学、安慰性语言或模糊占卜话术。

硬规则：
- 排盘与分析均已由程序精确算出（含真太阳时、月令三倍加权、十神分值、刑冲合害清单、格局与用神候选）。以用户消息里给出的「权威事实表」为准，不要自行重算，不要与它矛盾。
- 事实表里有的刑冲合害、十神、格局，你可以直接用；事实表里没有的，一律视作「此盘未见」，不得编造。
- 每条判断都要点名具体干支（例如「乙卯」）或具体十神（例如「月干正印乙」）或事实表里的某一条关系（例如「卯戌六合化火」）；一句话里至少一处落到具体命盘元素。
- 不出现任何古籍名、不写「出自某某经典」「古书云」之类字样；用户看不到出处，只看到结论和依据。
- 只推演已经过去的流年（不晚于「今天」所在年份）；对未来只谈大运方向，不断言具体事件。
- 回溯事件写成可以用「是 / 否」回答的具体陈述（哪一年、哪一类事），不要写成套话。
- 不预测生死、重病、官司输赢、具体投资结果。
- 语言：中文。

输出：只输出一个 JSON 对象，不要任何其他文字，键固定如下——
{
  "verdict": "四到十个字的判词，概括全局",
  "strength": {
    "level": "身强 / 身弱 / 中和 / 从强 / 从弱 / 从财格 / 从杀格 / 从儿格 / 从官格 等；以事实表给出的 level 为准",
    "reasons": ["两到四条依据，每条一句，点名干支或事实表中的某一项"]
  },
  "pattern": {"name": "格局名，与事实表的 pattern.candidates 一致", "explain": "两到三句：为何成格、成败关键"},
  "gods": {
    "use":   {"element": "用神五行", "why": "一句，点依据"},
    "like":  {"element": "喜神五行", "why": "一句"},
    "avoid": {"element": "忌神五行，可多个用顿号分隔", "why": "一句"},
    "idle":  {"element": "闲神五行，可为空字符串", "why": "一句，可为空字符串"}
  },
  "da_yun": [
    {"ganzhi": "与给定大运逐一对应，只写前六步", "theme": "四个字的主题", "reading": "两到三句：十神关系、与原局的刑冲合害、吉凶倾向"}
  ],
  "liu_nian": [
    {"year": 2016, "ganzhi": "丙申", "tag": "学业 / 家庭 / 情绪 / 关系 / 事业 / 迁移 / 健康 之一", "event": "一到两句：这一年多半发生了什么，以及命理依据"}
  ],
  "verify": ["四到六条可以用「是 / 否」回答的验证问题，带年份"],
  "summary": "三到五句总评：一生的主线、最该把握的十年、最该防的一点"
}
liu_nian 挑六到十个最关键的已过流年，按年份从早到晚排列。"""


# ── 把事实表写成模型可读的「权威输入」 ──────────────────────────
def _fact_lines(a: dict) -> list[str]:
    s = a['strength']
    g = a['gods']
    p = a['pattern']
    tg = a['ten_gods']
    br = a['relations']['branch']
    sg = a['relations']['stem']

    lines = [
        '【权威事实表（程序算出，不得矛盾）】',
        f"日主：{a['day_master']['gan']}{a['day_master']['element']}（月支{a['day_master']['month_zhi']}）。",
        f"强弱判定：{s['level']}；生扶力 {s['friendly_score']} vs 克泄耗力 {s['hostile_score']}；"
        f"得令={'是' if s['got_season'] else '否'}；有根={'是' if s['has_root'] else '否'}。",
        '强弱依据：' + '；'.join(s['reasons']),
    ]
    if s.get('follow_candidate') and (s['follow_candidate'] or {}).get('kind'):
        lines.append(f"从格候选：{s['follow_candidate']['kind']}（{s['follow_candidate']['reason']}）。")

    lines.append('五行力量（月令三倍加权百分比）：'
                 + '、'.join(f"{k}{v}%" for k, v in a['elements_percent'].items()))

    top5 = tg[:5]
    lines.append('十神力量排序（前五）：' + '；'.join(f"{t['star']}{t['score']}" for t in top5))

    rel_bits = []
    for name, zh in [('liu_he', '六合'), ('chong', '相冲'), ('hai', '相害'),
                     ('xing', '相刑'), ('zi_xing', '自刑'),
                     ('san_he', '三合'), ('ban_he', '半三合'), ('san_hui', '三会')]:
        if br.get(name):
            rel_bits.append(zh + '：' + '、'.join(r['detail'] + ('（涉日支）' if r.get('on_day') else '')
                                                 for r in br[name]))
    for name, zh in [('he', '天干五合'), ('ke', '天干相克')]:
        if sg.get(name):
            rel_bits.append(zh + '：' + '、'.join(r['detail'] + ('（涉日干）' if r.get('on_day') else '')
                                                 for r in sg[name]))
    if rel_bits:
        lines.append('关系清单：' + '；'.join(rel_bits))
    else:
        lines.append('关系清单：此盘无常见刑冲合害。')

    lines.append(f"格局候选：{p['name']}（{p['basis']}）"
                 + (f"；成格喜：{p['formed_hint']}" if p.get('formed_hint') else '')
                 + (f"；破格防：{p['broken_hint']}" if p.get('broken_hint') else ''))

    lines.append('用神候选：用=' + '、'.join(g['use'] or ['（未定）'])
                 + '；喜=' + '、'.join(g['like'] or ['（未定）'])
                 + '；忌=' + '、'.join(g['avoid'])
                 + '；闲=' + '、'.join(g['idle']))
    if g.get('notes'):
        lines.append('用神说明：' + '；'.join(g['notes']))

    hits = corpus.hits_for(a)
    if hits:
        lines.append('可用短语（不是出处，只是现成的说法，可挑拣用）：')
        for h in hits:
            lines.append(f"  - {h['headline']}：{h['body']}")

    return lines


# ── Pillar / chart description (unchanged shape) ─────────────────
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


# ── Build ────────────────────────────────────────────────────────
def build(chart: dict) -> list[dict]:
    a = analyze(chart)
    facts = '\n'.join(_fact_lines(a))
    user = (describe(chart) + '\n\n' + facts +
            "\n\n请完成：② 判断日主强弱、格局、用神、忌神（与事实表一致）；"
            "③ 排出大运与已走过的重要流年；"
            "④ 回溯并推演过去真实发生过的重要事件（学业、家庭、情绪转折、关系、事业节点），用于验证命盘。"
            "按规定的 JSON 格式输出。每句判断都要点名事实表里的某一条。")
    return [{'role': 'system', 'content': _SYSTEM + lessons('bazi')},
            {'role': 'user', 'content': user}]


# ── Validate the model's reply against the fact table ───────────
_STAR_NAMES = frozenset({'比肩', '劫财', '食神', '伤官', '正财', '偏财',
                         '正官', '七杀', '正印', '偏印'})
_STRENGTH_LEVELS = frozenset({'身强', '身弱', '中和', '从强', '从弱',
                              '从财格', '从杀格', '从官格', '从儿格'})


def validate(reading: dict, chart: dict) -> list[str]:
    """Return a list of plain-text contradiction notes. Empty = no issue."""
    if not reading or not isinstance(reading, dict):
        return []
    a = analyze(chart)
    notes: list[str] = []

    # 1. Strength label must match (adjacent 身强 / 中和 or 身弱 / 中和 is tolerated).
    want = a['strength']['level']
    got = (reading.get('strength') or {}).get('level', '').strip()
    if got and got not in _STRENGTH_LEVELS:
        notes.append(f"strength.level 「{got}」不是允许的强弱标签。")
    if got and want and got != want:
        # tolerance: 中和 ↔ 身强 / 身弱 允许靠近
        tolerated = {('身强', '中和'), ('中和', '身强'),
                     ('身弱', '中和'), ('中和', '身弱')}
        if (want, got) not in tolerated:
            notes.append(f"strength.level 「{got}」与事实表「{want}」矛盾。")

    # 2. Any ten-god names in gods.*.why / pattern.explain / da_yun / liu_nian must be real.
    all_text = ' '.join([
        (reading.get('pattern') or {}).get('explain', '') or '',
        ' '.join((v or {}).get('why', '') for v in (reading.get('gods') or {}).values()),
        ' '.join(x.get('reading', '') for x in (reading.get('da_yun') or []) if isinstance(x, dict)),
        ' '.join(x.get('event', '') for x in (reading.get('liu_nian') or []) if isinstance(x, dict)),
        reading.get('summary', '') or '',
    ])
    # Catch typos like 「正杀」「食印」 etc. by scanning for 2-char segments that look like a star name.
    bad_stars: set[str] = set()
    for prefix in ('正', '偏', '食', '伤', '七', '比', '劫'):
        import re as _re
        for m in _re.finditer(prefix + r'[^，。；：、\s\d]', all_text):
            name = m.group(0)
            # Only flag two-char names that resemble a 十神 but aren't one.
            if len(name) == 2 and name not in _STAR_NAMES:
                # Common safe prefixes that aren't 十神 (e.g. 正月、偏旁) — the 十神 context
                # is narrow enough that we only flag these three roots strictly.
                if name[0] in ('正', '偏', '七') and name[1] in '官杀财印神':
                    bad_stars.add(name)
    for bad in bad_stars:
        notes.append(f"十神名「{bad}」写错；允许的名称：比肩、劫财、食神、伤官、正财、偏财、正官、七杀、正印、偏印。")

    # 3. Any 刑冲合害 the model names must appear in the fact table.
    br = a['relations']['branch']
    sg = a['relations']['stem']
    valid_details: set[str] = set()
    for key in ('liu_he', 'chong', 'hai', 'xing', 'zi_xing', 'san_he', 'ban_he', 'san_hui'):
        for r in br.get(key, []):
            valid_details.add(r.get('detail', ''))
    for key in ('he', 'ke'):
        for r in sg.get(key, []):
            valid_details.add(r.get('detail', ''))

    import re as _re2
    # 关系格式：两个干支字符，可选「相」，最后「合 / 冲 / 刑 / 害 / 克」。
    claims = set()
    for m in _re2.finditer(
            r'([甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥]{2})相?(合|冲|刑|害|克)',
            all_text):
        claims.add((m.group(1), m.group(2)))
    for head, verb in claims:
        # If no fact detail contains the two chars together, it is fabricated.
        rev = head[::-1]
        if any((head in d) or (rev in d) for d in valid_details):
            continue
        notes.append(f"宣称的关系「{head}相{verb}」在事实表里未见。")

    return notes


def repair(reading: dict, contradictions: list[str]) -> dict:
    """When a retry still fails, drop or soften the offending lines so the
    output stays coherent. The schema remains intact; the user sees fewer
    words but no contradictions."""
    if not reading or not contradictions:
        return reading
    joined = ' | '.join(contradictions)
    # strength.level mismatch → fall back to a conservative 「中和」 label
    if 'strength.level' in joined:
        s = reading.setdefault('strength', {})
        s['level'] = '中和'
        s.setdefault('reasons', []).append('（原强弱标签与事实表矛盾，已改为保守判断）')
    # Fabricated relations → scrub from da_yun/reading, liu_nian/event, summary
    import re as _re
    for claim_match in _re.finditer(r"「([^」]+)」在事实表里未见", joined):
        bad = claim_match.group(1)
        for section in ('da_yun', 'liu_nian'):
            for item in reading.get(section, []) or []:
                key = 'reading' if section == 'da_yun' else 'event'
                if isinstance(item, dict) and bad in (item.get(key) or ''):
                    item[key] = (item[key] or '').replace(bad, '').strip()
        if reading.get('summary') and bad in reading['summary']:
            reading['summary'] = reading['summary'].replace(bad, '').strip()
    return reading


# ── Parse (unchanged shape) ──────────────────────────────────────
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

FOLLOWUP_VERSION = 'bazi-ask-v2'   # unchanged (scope rules from scope.md, 2026-10-12)
MAX_QUESTION_CHARS = 300
HISTORY_TURNS = 8

# 追问系统提示词不注入事实表（工具链路上由 build_followup 单独负责），
# 置换掉划掉的关于古籍的那句话，换成角色的纯述。
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
