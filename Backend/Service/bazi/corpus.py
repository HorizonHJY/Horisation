"""A small rule table: fact → short classical phrase.

Not a dataset. Not citations from a book — those cannot appear on screen and
the model is told not to say them. These are just the common two-sentence
summaries attached to the most common bazi configurations: 伤官见官、食神
制杀、财星坏印、日支被冲、月令被合、各格成破、从格顺逆. The analysis
module computes the condition; this module names the condition in a short
line, so the prompt has something concrete to echo back instead of
freewheeling.

Written in plain modern Chinese. Where a classical four-character tag is
unambiguous and in common use (e.g. 「伤官见官」) it is kept as the headline;
the body stays plain.

Each entry: {condition_key, headline, body}. The matcher sits at the bottom.
"""

from __future__ import annotations

from typing import Callable


# ── 十神组合（看命盘里的 ten_gods 排序和 pattern） ─────────────────
STAR_COMBOS: list[dict] = [
    {'key': 'shang_guan_jian_guan',
     'headline': '伤官见官',
     'body': '伤官与正官同现，口舌、名位、与上位者的冲突容易偏重；若无印来调和，意气常起波澜。'},
    {'key': 'shi_shen_zhi_sha',
     'headline': '食神制杀',
     'body': '食神与七杀同见，食神能驯服七杀的冲劲，压力转为可控的事功。'},
    {'key': 'cai_xing_huai_yin',
     'headline': '财星坏印',
     'body': '正偏财与印同透，财动则印难安，物欲与涵养常相拉扯；重在守住印之本。'},
    {'key': 'guan_yin_xiang_sheng',
     'headline': '官印相生',
     'body': '官星与印同透，名位与涵养相互滋养，适合在建制内积累。'},
    {'key': 'sha_yin_xiang_sheng',
     'headline': '杀印相生',
     'body': '七杀与印同透，压力推动学习与定力，适合技术、专业、军警司法一路。'},
    {'key': 'shi_shen_sheng_cai',
     'headline': '食神生财',
     'body': '食神与财同见，才情能换成实利，适合靠一技之长变现。'},
    {'key': 'shang_guan_pei_yin',
     'headline': '伤官配印',
     'body': '伤官有印来约束，才气不再外泄为冲撞，宜学宜教。'},
    {'key': 'bi_jie_duo_cai',
     'headline': '比劫夺财',
     'body': '比劫叠见而财弱，钱财易被同类分走；合伙、兄弟之间常有金钱口角。'},
    {'key': 'yin_duo_shen_qiang',
     'headline': '印多身强',
     'body': '印星过旺而日主得助，易懒、依赖、拖延；需食伤或财来发之。'},
    {'key': 'sha_wang_wu_zhi',
     'headline': '杀旺无制',
     'body': '七杀过旺而无食神印星制化，压力转为内耗与冲动，宜寻制杀之道。'},
]


def _has(star: str, ten_gods: list[dict]) -> bool:
    return any(t['star'] == star and t['score'] > 0 for t in ten_gods)


def _top(ten_gods: list[dict], n: int = 3) -> list[str]:
    return [t['star'] for t in ten_gods[:n]]


def _star_combo_matches(ten_gods: list[dict]) -> list[dict]:
    hits = []
    top3 = set(_top(ten_gods, 3))

    def pair(a: str, b: str) -> bool:
        return _has(a, ten_gods) and _has(b, ten_gods)

    if pair('伤官', '正官'):
        hits.append('shang_guan_jian_guan')
    if pair('食神', '七杀'):
        hits.append('shi_shen_zhi_sha')
    if (pair('正财', '正印') or pair('偏财', '正印') or
            pair('正财', '偏印') or pair('偏财', '偏印')):
        hits.append('cai_xing_huai_yin')
    if pair('正官', '正印') or pair('正官', '偏印'):
        hits.append('guan_yin_xiang_sheng')
    if pair('七杀', '正印') or pair('七杀', '偏印'):
        hits.append('sha_yin_xiang_sheng')
    if pair('食神', '正财') or pair('食神', '偏财'):
        hits.append('shi_shen_sheng_cai')
    if pair('伤官', '正印') or pair('伤官', '偏印'):
        hits.append('shang_guan_pei_yin')
    if (pair('比肩', '正财') or pair('比肩', '偏财') or
            pair('劫财', '正财') or pair('劫财', '偏财')) and ({'比肩', '劫财'} & top3):
        hits.append('bi_jie_duo_cai')
    if ({'正印', '偏印'} & top3) and _has('比肩', ten_gods):
        hits.append('yin_duo_shen_qiang')
    if ('七杀' in top3) and not pair('七杀', '食神') and not (pair('七杀', '正印') or pair('七杀', '偏印')):
        hits.append('sha_wang_wu_zhi')

    out = []
    for key in hits:
        row = next((r for r in STAR_COMBOS if r['key'] == key), None)
        if row:
            out.append(row)
    return out


# ── 刑冲合害（看日支与月支） ────────────────────────────────────────
RELATION_RULES: list[dict] = [
    {'key': 'ri_chong',
     'headline': '日支被冲',
     'body': '日支为夫妻宫，相冲则家宅、婚姻、合伙中容易起摩擦与变动。'},
    {'key': 'yue_he',
     'headline': '月令被合',
     'body': '月令为提纲，被合化则格局气象改变，职业轨迹容易转向。'},
    {'key': 'yue_chong',
     'headline': '月令受冲',
     'body': '月令受冲，提纲动摇，立身的那条线不稳。'},
    {'key': 'san_he_cheng_ju',
     'headline': '三合成局',
     'body': '三支齐见成合，所成之五行力量大增；注意顺应其势。'},
    {'key': 'san_hui_cheng_fang',
     'headline': '三会成方',
     'body': '三会齐见成方，所会五行独盛，气象偏向一方。'},
    {'key': 'ri_zhi_zi_xing',
     'headline': '日支自刑',
     'body': '日支与另一支同字，内耗偏重，易自我纠结。'},
    {'key': 'ri_zhi_hai',
     'headline': '日支被害',
     'body': '日支相害，与配偶、合伙人之间多有隐性摩擦。'},
    {'key': 'san_xing_quan_bei',
     'headline': '三刑全备',
     'body': '寅巳申或丑戌未三刑齐见，官司、意外、情绪拉扯的压力偏大。'},
]


def _relation_matches(relations: dict) -> list[dict]:
    hits: list[str] = []
    br = relations['branch']
    if any(r.get('on_day') for r in br['chong']):
        hits.append('ri_chong')
    if any(r.get('on_day') for r in br['hai']):
        hits.append('ri_zhi_hai')
    if any(r.get('on_day') for r in br['zi_xing']):
        hits.append('ri_zhi_zi_xing')
    if any('month' in r.get('at', []) for r in br['liu_he']):
        hits.append('yue_he')
    if any('month' in r.get('at', []) for r in br['chong']):
        hits.append('yue_chong')
    if br['san_he']:
        hits.append('san_he_cheng_ju')
    if br['san_hui']:
        hits.append('san_hui_cheng_fang')
    for r in br['xing']:
        if '三刑' in r.get('detail', ''):
            hits.append('san_xing_quan_bei')
            break
    out = []
    for key in hits:
        row = next((r for r in RELATION_RULES if r['key'] == key), None)
        if row:
            out.append(row)
    return out


# ── 格局成败（看 pattern + ten_gods） ──────────────────────────────
PATTERN_RULES: list[dict] = [
    {'key': 'zheng_guan_cheng',
     'match': lambda a: a['pattern']['name'] == '正官格' and _has('正印', a['ten_gods']) and _has('正财', a['ten_gods']),
     'headline': '正官格有财印',
     'body': '财生官、印护官，正官格最标准的助力结构。'},
    {'key': 'zheng_guan_po',
     'match': lambda a: a['pattern']['name'] == '正官格' and _has('伤官', a['ten_gods']),
     'headline': '正官格见伤官',
     'body': '伤官见官格,利需有印来调和,否则格局偏破。'},
    {'key': 'qi_sha_zhi',
     'match': lambda a: a['pattern']['name'] == '七杀格' and _has('食神', a['ten_gods']),
     'headline': '七杀格有食神',
     'body': '食神制杀，压力有度，成就偏向专业与攻坚。'},
    {'key': 'qi_sha_wang_zhi',
     'match': lambda a: a['pattern']['name'] == '七杀格' and not _has('食神', a['ten_gods']) and not _has('正印', a['ten_gods']) and not _has('偏印', a['ten_gods']),
     'headline': '七杀无制',
     'body': '七杀格无食神无印，煞气直冲，压力难以消化。'},
    {'key': 'shi_shen_cheng',
     'match': lambda a: a['pattern']['name'] == '食神格' and (_has('正财', a['ten_gods']) or _has('偏财', a['ten_gods'])),
     'headline': '食神生财',
     'body': '食神格财星透出，才气有去处，成格可观。'},
    {'key': 'shi_shen_po',
     'match': lambda a: a['pattern']['name'] == '食神格' and _has('偏印', a['ten_gods']),
     'headline': '偏印夺食',
     'body': '食神格见偏印，才气被夺，宜以财解印。'},
    {'key': 'zheng_yin_cheng',
     'match': lambda a: a['pattern']['name'] == '正印格' and _has('正官', a['ten_gods']),
     'headline': '官印相生',
     'body': '正印格见正官，官来生印，正印格最佳结构。'},
    {'key': 'zheng_yin_po',
     'match': lambda a: a['pattern']['name'] == '正印格' and (_has('正财', a['ten_gods']) or _has('偏财', a['ten_gods'])),
     'headline': '财破印',
     'body': '正印格见财星，财动则印摇，宜以比劫化之。'},
    {'key': 'jian_lu',
     'match': lambda a: a['pattern']['name'] in ('建禄格', '月劫格'),
     'headline': '建禄 / 月劫格',
     'body': '日主得月令本气，身自强；喜食伤官杀财疏泄之，忌比劫叠见无制。'},
    {'key': 'cong_cai',
     'match': lambda a: (a['strength'].get('follow_candidate') or {}).get('kind') == '从财格',
     'headline': '从财格',
     'body': '日主无根，财势独大，顺之则吉；忌比劫印来破格。'},
    {'key': 'cong_sha',
     'match': lambda a: (a['strength'].get('follow_candidate') or {}).get('kind') == '从杀格',
     'headline': '从杀格',
     'body': '日主无根，官杀当令，顺从其势；忌食伤印来犯格。'},
    {'key': 'cong_er',
     'match': lambda a: (a['strength'].get('follow_candidate') or {}).get('kind') == '从儿格',
     'headline': '从儿格',
     'body': '日主无根，食伤独盛，顺之以成才气外化；忌印来破。'},
]


def _pattern_matches(analysis: dict) -> list[dict]:
    out = []
    for rule in PATTERN_RULES:
        try:
            if rule['match'](analysis):
                out.append({'key': rule['key'], 'headline': rule['headline'], 'body': rule['body']})
        except Exception:
            # Rule errors never block a reading; the rule is just skipped.
            continue
    return out


# ── 入口 ─────────────────────────────────────────────────────────────
def hits_for(analysis: dict) -> list[dict]:
    """命中条目，供 prompt 作为「可用的现成短语库」。
    返回列表最多裁到 10 条，避免 prompt 过长。
    """
    all_hits = (_star_combo_matches(analysis['ten_gods'])
                + _relation_matches(analysis['relations'])
                + _pattern_matches(analysis))
    # 保序去重
    seen: set[str] = set()
    unique = []
    for h in all_hits:
        key = h.get('key') or h['headline']
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
    return unique[:10]


def count() -> int:
    """供文档 / 测试报告使用：当前语料总条数。"""
    return len(STAR_COMBOS) + len(RELATION_RULES) + len(PATTERN_RULES)
