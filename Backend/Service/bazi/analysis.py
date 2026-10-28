"""The BaZi analysis — turn the arithmetic chart into a table of facts.

chart.py already computes the four pillars, hidden stems, visible ten gods,
growth stages, voids, luck pillars and a weighted five-element tally. What it
does not do is interpret: it does not say whether the day master is strong,
which god to use, which pattern the month branch forms, which branches clash.
Those have been the model's job, and the model is loud and wrong often
enough to need help.

This module adds the layer between. Every number here is reproducible: no
"intuition", no classical poetry — just the standard weights (月令三倍、
藏干 1 / 0.5 / 0.3), the standard relation tables (三合 / 六合 / 六冲 /
相害 / 相刑 / 自刑), and the standard way the four 月令 branches are read
for a pattern. The output is a flat dict shaped to be dropped straight into
the prompt, and to be checked against the model's reply afterwards.

Nothing here uses external data. The classical rules are public domain and
can be found in any 子平 textbook; the specific weights are the common ones.
"""

from __future__ import annotations

from .chart import GAN, ZHI, GAN_EL, ZHI_EL, ELEMENTS

# ── 关系表（公版） ───────────────────────────────────────────────────
_SHENG = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
_KE = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}

# 天干五合 / 相克
GAN_HE = {
    frozenset('甲己'): '土', frozenset('乙庚'): '金',
    frozenset('丙辛'): '水', frozenset('丁壬'): '木',
    frozenset('戊癸'): '火',
}
GAN_KE = {('甲', '戊'), ('乙', '己'), ('丙', '庚'), ('丁', '辛'), ('戊', '壬'),
          ('己', '癸'), ('庚', '甲'), ('辛', '乙'), ('壬', '丙'), ('癸', '丁')}

# 地支关系
ZHI_LIU_HE = {
    frozenset('子丑'): '土', frozenset('寅亥'): '木', frozenset('卯戌'): '火',
    frozenset('辰酉'): '金', frozenset('巳申'): '水', frozenset('午未'): '土',
}
ZHI_CHONG = [frozenset(p) for p in ('子午', '丑未', '寅申', '卯酉', '辰戌', '巳亥')]
ZHI_HAI = [frozenset(p) for p in ('子未', '丑午', '寅巳', '卯辰', '申亥', '酉戌')]
ZHI_SAN_HE = {
    frozenset('申子辰'): '水', frozenset('亥卯未'): '木',
    frozenset('寅午戌'): '火', frozenset('巳酉丑'): '金',
}
# 半三合：三合中带「子/午/卯/酉」中气者才算半合
_SAN_HE_PIVOT = {frozenset('申子辰'): '子', frozenset('亥卯未'): '卯',
                 frozenset('寅午戌'): '午', frozenset('巳酉丑'): '酉'}
# 三会
ZHI_SAN_HUI = {
    frozenset('寅卯辰'): '木', frozenset('巳午未'): '火',
    frozenset('申酉戌'): '金', frozenset('亥子丑'): '水',
}
# 三刑：寅巳申、丑戌未；相刑：子卯；自刑：辰辰、午午、酉酉、亥亥
_SAN_XING = (('寅', '巳', '申'), ('丑', '戌', '未'))
_XIANG_XING = frozenset('子卯')
_ZI_XING = ('辰', '午', '酉', '亥')

# 阴阳：十干偶数位（0,2,4,6,8）为阳
GAN_YANG = {g: (i % 2 == 0) for i, g in enumerate(GAN)}


# ── 十神关系 ─────────────────────────────────────────────────────────
def ten_god(me: str, other: str) -> str:
    """以 me 为日主，other 这个天干对我的十神。"""
    me_el, other_el = GAN_EL[me], GAN_EL[other]
    same_yy = GAN_YANG[me] == GAN_YANG[other]
    if me_el == other_el:
        return '比肩' if same_yy else '劫财'
    if _SHENG[me_el] == other_el:
        return '食神' if same_yy else '伤官'
    if _KE[me_el] == other_el:
        return '偏财' if same_yy else '正财'
    if _KE[other_el] == me_el:
        return '七杀' if same_yy else '正官'
    return '偏印' if same_yy else '正印'


# 十神分类：生扶（印、比劫）vs 克泄耗（官杀、食伤、财）
_FRIENDS = ('比肩', '劫财', '正印', '偏印')
_FOES = ('食神', '伤官', '正财', '偏财', '正官', '七杀')


def is_friend(star: str) -> bool:
    return star in _FRIENDS


# ── 五行加权 ─────────────────────────────────────────────────────────
# 透出天干 1；藏干本气 1、中气 0.5、余气 0.3（与 chart.py 一致）
# 月令三倍：描述力量时月令地支的所有藏干 ×3
_HIDDEN_WEIGHTS = (1.0, 0.5, 0.3)


def _weighted_elements(pillars: list[dict]) -> dict:
    """月令三倍加权的五行力量。
    这是判断得令、找格局的基础；画面上的柱状图仍用 chart.py 的平权版本。
    """
    score = {e: 0.0 for e in ELEMENTS}
    for i, p in enumerate(pillars):
        multiplier = 3.0 if i == 1 else 1.0
        score[GAN_EL[p['gan']]] += 1.0 * multiplier
        for w, h in zip(_HIDDEN_WEIGHTS, p['hidden']):
            score[h['element']] += w * multiplier
    return {e: round(v, 2) for e, v in score.items()}


def _element_percentages(score: dict) -> dict:
    total = sum(score.values()) or 1.0
    return {e: round(v / total * 100, 1) for e, v in score.items()}


# ── 日主强弱 ─────────────────────────────────────────────────────────
# 三项评分：得令 / 得地 / 得势，然后生扶 vs 克泄耗。
# 判级依据对两边比值做阈值划分，阈值取 0.7 / 1.4，经验值可调。

# 月令地支对日主的作用：命中则加分，否则扣分
# 规则：以月支本气为准
#   - 本气 = 比劫 → +3（得令、建禄 / 月劫）
#   - 本气 = 正偏印 → +2.5（印旺）
#   - 本气 = 食伤 → -2（泄）
#   - 本气 = 财 → -2.5（耗）
#   - 本气 = 官杀 → -3（克）
# 中气按一半、余气三分之一

def _month_support(day_gan: str, month_pillar: dict) -> dict:
    """月令对日主的作用分（三组权重：本气 / 中气 / 余气）。"""
    weights = _HIDDEN_WEIGHTS
    scores = []
    for w, h in zip(weights, month_pillar['hidden']):
        star = ten_god(day_gan, h['gan'])
        v = {
            '比肩': 3.0, '劫财': 3.0,
            '正印': 2.5, '偏印': 2.5,
            '食神': -2.0, '伤官': -2.0,
            '正财': -2.5, '偏财': -2.5,
            '正官': -3.0, '七杀': -3.0,
        }[star]
        scores.append({'gan': h['gan'], 'star': star, 'weight': w, 'score': round(v * w, 2)})
    total = round(sum(s['score'] for s in scores), 2)
    return {'cells': scores, 'total': total, 'got_season': total > 0}


def _branch_root(day_gan: str, pillars: list[dict]) -> dict:
    """得地：日主在四柱地支里是否有根（比劫 / 印 在藏干中）。
    本气根最强，中气次之，余气最弱；每命中一处记录。
    """
    hits = []
    for i, p in enumerate(pillars):
        for w, h in zip(_HIDDEN_WEIGHTS, p['hidden']):
            star = ten_god(day_gan, h['gan'])
            if star in _FRIENDS:
                hits.append({
                    'pillar': p['label'], 'zhi': p['zhi'], 'gan': h['gan'],
                    'star': star, 'weight': w,
                    'kind': '本气' if w == 1.0 else ('中气' if w == 0.5 else '余气'),
                })
    strong_root = any(h['weight'] == 1.0 and h['star'] in ('比肩', '劫财') for h in hits)
    return {'hits': hits, 'count': len(hits),
            'strong_root': strong_root,
            'score': round(sum(h['weight'] * (2.0 if h['star'] in ('比肩', '劫财') else 1.5) for h in hits), 2)}


def _stem_support(day_gan: str, pillars: list[dict]) -> dict:
    """得势：年 / 月 / 时三个透干对日主的作用（日主本身不计）。"""
    hits = []
    for i, p in enumerate(pillars):
        if i == 2:
            continue
        star = ten_god(day_gan, p['gan'])
        hits.append({'pillar': p['label'], 'gan': p['gan'], 'star': star})
    friendly = sum(1 for h in hits if h['star'] in _FRIENDS)
    hostile = sum(1 for h in hits if h['star'] in _FOES)
    score = 0.0
    for h in hits:
        s = h['star']
        if s in ('比肩', '劫财'):
            score += 1.5
        elif s in ('正印', '偏印'):
            score += 1.2
        elif s in ('食神', '伤官'):
            score -= 0.8
        elif s in ('正财', '偏财'):
            score -= 1.0
        else:
            score -= 1.5
    return {'hits': hits, 'friendly': friendly, 'hostile': hostile, 'score': round(score, 2)}


def _strength(day_gan: str, pillars: list[dict]) -> dict:
    """身强 / 身弱 / 中和 / 从格候选的判断。
    身强 / 身弱靠「三项评分」加总，和「失令」时的从格候选并行给。
    """
    season = _month_support(day_gan, pillars[1])
    root = _branch_root(day_gan, pillars)
    stems = _stem_support(day_gan, pillars)

    # 综合分：月令三项分合起来视为「生扶力」或「克泄耗力」的决定项
    friendly = max(season['total'], 0) + root['score'] + max(stems['score'], 0)
    hostile = max(-season['total'], 0) + max(-stems['score'], 0)
    # 月令非友方时，根与透干的生扶要先打折（印比在失令时效率下降）
    if not season['got_season']:
        friendly *= 0.75

    ratio = friendly / hostile if hostile > 1e-6 else float('inf')
    if root['count'] == 0 and not season['got_season'] and friendly < 1.5:
        # 日主无根、失令，且生扶几乎为零 → 从格候选
        follow = _follow_candidate(day_gan, pillars, stems)
        level = follow['kind'] or '身弱'
        follow_hint = follow
    elif ratio >= 1.4:
        level = '身强'
        follow_hint = None
    elif ratio <= 0.7:
        level = '身弱'
        follow_hint = None
    else:
        level = '中和'
        follow_hint = None

    return {
        'level': level,
        'ratio': round(ratio, 2) if ratio != float('inf') else None,
        'friendly_score': round(friendly, 2),
        'hostile_score': round(hostile, 2),
        'got_season': season['got_season'],
        'season_breakdown': season,
        'has_root': root['count'] > 0,
        'root_breakdown': root,
        'stem_breakdown': stems,
        'follow_candidate': follow_hint,
        'reasons': _strength_reasons(level, season, root, stems),
    }


def _strength_reasons(level: str, season: dict, root: dict, stems: dict) -> list[str]:
    """用一句话一条，给出判断依据，供提示词 / 校验层使用。"""
    out = []
    if season['got_season']:
        got = [c for c in season['cells'] if c['score'] > 0]
        if got:
            out.append(f"月令得令：{'、'.join(c['gan'] + c['star'] for c in got)}")
    else:
        lost = [c for c in season['cells'] if c['score'] < 0]
        if lost:
            out.append(f"月令失令：{'、'.join(c['gan'] + c['star'] for c in lost)}")
    if root['count'] > 0:
        strong = [h for h in root['hits'] if h['weight'] == 1.0]
        if strong:
            out.append(f"地支有本气根：{'、'.join(h['pillar'] + '-' + h['gan'] + h['star'] for h in strong)}")
        else:
            out.append(f"地支仅有藏干虚根（{root['count']} 处）")
    else:
        out.append('四支皆无比劫印之根')
    if stems['friendly'] > stems['hostile']:
        out.append(f"透干助我（{stems['friendly']} 处生扶 vs {stems['hostile']} 处克泄）")
    elif stems['hostile'] > stems['friendly']:
        out.append(f"透干克我（{stems['hostile']} 处克泄 vs {stems['friendly']} 处生扶）")
    return out


def _follow_candidate(day_gan: str, pillars: list[dict], stems: dict) -> dict:
    """从格候选：日主无根失令时，看哪种五行一方独大，从之。"""
    # 看非日主透干中哪一类十神占优
    tally: dict[str, int] = {}
    for p in pillars:
        for h in p['hidden']:
            star = ten_god(day_gan, h['gan'])
            if star in _FRIENDS:
                continue
            tally[star] = tally.get(star, 0) + 1
    if not tally:
        return {'kind': None, 'reason': ''}
    dominant = max(tally.items(), key=lambda kv: kv[1])
    name_map = {
        '正官': '从官格', '七杀': '从杀格',
        '正财': '从财格', '偏财': '从财格',
        '食神': '从儿格', '伤官': '从儿格',
    }
    kind = name_map.get(dominant[0])
    return {'kind': kind, 'dominant_star': dominant[0], 'count': dominant[1],
            'reason': f'日主无根，{dominant[0]} 在四柱藏干中出现 {dominant[1]} 次'}


# ── 十神力量排序 ─────────────────────────────────────────────────────
def _ten_gods_ranking(day_gan: str, pillars: list[dict]) -> list[dict]:
    """十神按加权力量从大到小排序。月令地支三倍。"""
    tally: dict[str, float] = {}
    source: dict[str, list[str]] = {}
    for i, p in enumerate(pillars):
        mul = 3.0 if i == 1 else 1.0
        if i != 2:   # 日干不计十神
            star = ten_god(day_gan, p['gan'])
            tally[star] = tally.get(star, 0) + 1.0 * mul
            source.setdefault(star, []).append(f"{p['label']}-天干{p['gan']}")
        for w, h in zip(_HIDDEN_WEIGHTS, p['hidden']):
            star = ten_god(day_gan, h['gan'])
            tally[star] = tally.get(star, 0) + w * mul
            source.setdefault(star, []).append(f"{p['label']}-藏干{h['gan']}")
    return sorted(
        [{'star': s, 'score': round(v, 2), 'sources': source[s]} for s, v in tally.items()],
        key=lambda d: -d['score'],
    )


# ── 刑冲合害 ─────────────────────────────────────────────────────────
PILLAR_KEYS = ('year', 'month', 'day', 'time')
PILLAR_LABEL = {'year': '年', 'month': '月', 'day': '日', 'time': '时'}


def _branch_relations(pillars: list[dict]) -> dict:
    """列出所有地支之间的刑冲合害、三合、半合、三会、自刑。
    每条记录位置（年 / 月 / 日 / 时）与是否涉及日支，供校验层核对。"""
    out = {'liu_he': [], 'chong': [], 'hai': [], 'xing': [],
           'san_he': [], 'ban_he': [], 'san_hui': [], 'zi_xing': []}
    zhis = [(PILLAR_KEYS[i], p['zhi']) for i, p in enumerate(pillars)]

    # 两两组合
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = zhis[i], zhis[j]
            za, zb = a[1], b[1]
            pair = frozenset((za, zb))
            on_day = 'day' in (a[0], b[0])
            pos = f"{PILLAR_LABEL[a[0]]}支{za} × {PILLAR_LABEL[b[0]]}支{zb}"
            if za != zb and pair in ZHI_LIU_HE:
                out['liu_he'].append({'pos': pos, 'detail': f"{za}{zb}六合化{ZHI_LIU_HE[pair]}",
                                      'element': ZHI_LIU_HE[pair], 'on_day': on_day,
                                      'at': [a[0], b[0]]})
            if pair in ZHI_CHONG:
                out['chong'].append({'pos': pos, 'detail': f"{za}{zb}相冲",
                                     'on_day': on_day, 'at': [a[0], b[0]]})
            if pair in ZHI_HAI:
                out['hai'].append({'pos': pos, 'detail': f"{za}{zb}相害",
                                   'on_day': on_day, 'at': [a[0], b[0]]})
            if za != zb and (pair == _XIANG_XING):
                out['xing'].append({'pos': pos, 'detail': f"{za}{zb}子卯相刑",
                                    'on_day': on_day, 'at': [a[0], b[0]]})

    # 自刑：成对才算（例如两个辰）
    for z in _ZI_XING:
        pillars_at = [PILLAR_KEYS[i] for i, p in enumerate(pillars) if p['zhi'] == z]
        if len(pillars_at) >= 2:
            out['zi_xing'].append({'zhi': z, 'count': len(pillars_at), 'at': pillars_at,
                                   'on_day': 'day' in pillars_at,
                                   'detail': f"{z}{z}自刑"})

    # 三刑（寅巳申、丑戌未）
    present = [p['zhi'] for p in pillars]
    for triple in _SAN_XING:
        if all(z in present for z in triple):
            at = [PILLAR_KEYS[i] for i, p in enumerate(pillars) if p['zhi'] in triple]
            out['xing'].append({'detail': ''.join(triple) + '三刑', 'at': at,
                                'on_day': 'day' in at,
                                'pos': '、'.join(PILLAR_LABEL[k] + '支' for k in at)})

    # 三合、半合
    for group, elem in ZHI_SAN_HE.items():
        hit_pillars = [PILLAR_KEYS[i] for i, p in enumerate(pillars) if p['zhi'] in group]
        hit_zhis = [p['zhi'] for p in pillars if p['zhi'] in group]
        if len(set(hit_zhis)) == 3:
            out['san_he'].append({'detail': ''.join(group) + f'三合{elem}局',
                                  'element': elem, 'at': hit_pillars,
                                  'on_day': 'day' in hit_pillars})
        elif len(set(hit_zhis)) == 2 and _SAN_HE_PIVOT[group] in hit_zhis:
            out['ban_he'].append({'detail': ''.join(sorted(set(hit_zhis),
                                                           key=lambda z: list(group).index(z))) + f'半合{elem}',
                                  'element': elem, 'at': hit_pillars,
                                  'on_day': 'day' in hit_pillars})

    # 三会
    for group, elem in ZHI_SAN_HUI.items():
        hit_pillars = [PILLAR_KEYS[i] for i, p in enumerate(pillars) if p['zhi'] in group]
        hit_zhis = set(p['zhi'] for p in pillars if p['zhi'] in group)
        if len(hit_zhis) == 3:
            out['san_hui'].append({'detail': ''.join(group) + f'三会{elem}方',
                                   'element': elem, 'at': hit_pillars,
                                   'on_day': 'day' in hit_pillars})

    return out


def _stem_relations(pillars: list[dict]) -> dict:
    """天干五合、相克，列出所有两两关系。"""
    out = {'he': [], 'ke': []}
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = pillars[i]['gan'], pillars[j]['gan']
            label = f"{PILLAR_LABEL[PILLAR_KEYS[i]]}干{a} × {PILLAR_LABEL[PILLAR_KEYS[j]]}干{b}"
            on_day = 2 in (i, j)
            if a != b and frozenset((a, b)) in GAN_HE:
                out['he'].append({'detail': f"{a}{b}合化{GAN_HE[frozenset((a, b))]}",
                                  'element': GAN_HE[frozenset((a, b))],
                                  'at': [PILLAR_KEYS[i], PILLAR_KEYS[j]],
                                  'on_day': on_day, 'pos': label})
            if (a, b) in GAN_KE:
                out['ke'].append({'detail': f"{a}克{b}", 'at': [PILLAR_KEYS[i], PILLAR_KEYS[j]],
                                  'on_day': on_day, 'pos': label})
            if (b, a) in GAN_KE:
                out['ke'].append({'detail': f"{b}克{a}", 'at': [PILLAR_KEYS[j], PILLAR_KEYS[i]],
                                  'on_day': on_day, 'pos': label})
    return out


# ── 用神与格局 ─────────────────────────────────────────────────────
# 用神候选规则（扶抑用神）：
#   身强 → 克 / 泄 / 耗 的那一方（官杀 / 食伤 / 财）为用，三者里取力量最弱却可用的那一个
#   身弱 → 比劫 / 印 为用（先看哪类更欠缺）
#   中和 → 以调候 + 月令格局喜用为主
# 从格的用神是「被从者」本身。

def _usable_gods(day_gan: str, strength: dict, elem_pct: dict, ten_gods: list[dict],
                 month_zhi: str) -> dict:
    """用神 / 喜神 / 忌神 / 闲神候选。给候选名单与一句理由，不代替模型下结论。"""
    me_el = GAN_EL[day_gan]
    level = strength['level']
    friends_el = {me_el, _SHENG_INV(me_el)}   # 比劫 + 印 的五行
    foes_el = {_SHENG[me_el], _KE[me_el], _KE_INV(me_el)}  # 食伤 + 财 + 官杀

    use: list[str] = []
    avoid: list[str] = []
    notes: list[str] = []

    if level in ('身强', '中和'):
        # 克泄耗三选一：取在局中已出、力量弱而未成病者
        order = sorted(foes_el, key=lambda e: elem_pct.get(e, 0))
        use = [e for e in order if elem_pct.get(e, 0) > 0] or list(foes_el)
        avoid = sorted(friends_el, key=lambda e: -elem_pct.get(e, 0))
        notes.append(f"{level}：取{_KE_INV(me_el)}（官杀）/ {_SHENG[me_el]}（食伤）/ {_KE[me_el]}（财）为用，忌{me_el}（比劫）{_SHENG_INV(me_el)}（印）过旺")
    elif level == '身弱':
        order = sorted(friends_el, key=lambda e: elem_pct.get(e, 0))
        use = [e for e in order if elem_pct.get(e, 0) > 0] or list(friends_el)
        avoid = sorted(foes_el, key=lambda e: -elem_pct.get(e, 0))
        notes.append(f"身弱：取{me_el}（比劫）/ {_SHENG_INV(me_el)}（印）为用，忌{_KE_INV(me_el)}（官杀）{_SHENG[me_el]}（食伤）{_KE[me_el]}（财）")
    elif level and level.startswith('从'):
        # 从格：从者为用，反之为忌
        follow = strength.get('follow_candidate') or {}
        star = follow.get('dominant_star', '')
        star_el = {
            '正官': _KE_INV(me_el), '七杀': _KE_INV(me_el),
            '正财': _KE[me_el], '偏财': _KE[me_el],
            '食神': _SHENG[me_el], '伤官': _SHENG[me_el],
        }.get(star)
        if star_el:
            use = [star_el, _SHENG.get(star_el) or '', _KE.get(star_el) or '']
            use = [e for e in use if e]
            avoid = [me_el, _SHENG_INV(me_el)]
            notes.append(f"{level}：顺从{star}之势，忌比劫印来破格")

    # 调候：极寒极热时额外提醒
    climate = _climate_hint(me_el, month_zhi)
    if climate:
        notes.append(climate)

    like_el = [e for e in use[1:2]]      # 喜神：助用神的那一个（简化为用神的生者或同类）
    use_el = use[:1]
    idle_el = [e for e in ELEMENTS if e not in use and e not in avoid]
    return {
        'use': use_el, 'like': like_el, 'avoid': avoid, 'idle': idle_el,
        'notes': notes,
    }


def _SHENG_INV(e: str) -> str:
    """谁生 e。"""
    for k, v in _SHENG.items():
        if v == e:
            return k
    return ''


def _KE_INV(e: str) -> str:
    """谁克 e。"""
    for k, v in _KE.items():
        if v == e:
            return k
    return ''


def _climate_hint(day_el: str, month_zhi: str) -> str:
    """调候短提示：仅覆盖最明显的几种。"""
    winter = {'亥', '子', '丑'}
    summer = {'巳', '午', '未'}
    if month_zhi in winter:
        if day_el in ('火', '土'):
            return '调候：生于冬月，日主火土宜见火暖'
        if day_el == '水':
            return '调候：水生水月，宜见戊土制、丙火暖'
    if month_zhi in summer:
        if day_el in ('金', '水'):
            return '调候：生于夏月，日主金水宜见水润'
        if day_el == '火':
            return '调候：火生火月，宜见壬癸调候'
    return ''


# ── 格局（月令取格） ─────────────────────────────────────────────────
_PATTERN_BY_STAR = {
    '正官': '正官格', '七杀': '七杀格',
    '正财': '正财格', '偏财': '偏财格',
    '食神': '食神格', '伤官': '伤官格',
    '正印': '正印格', '偏印': '偏印格',
}


def _pattern(day_gan: str, pillars: list[dict]) -> dict:
    """月令取格：
      - 月令本气为比劫 → 建禄格（阳日主）或 月劫格（阴日主）
      - 其余：月令本气透天干定格；本气不透则看中气透出；再不透看余气
      - 给出成格 / 破格关键的候选说明（但不下定论）
    """
    month = pillars[1]
    other_pillars = [p for i, p in enumerate(pillars) if i != 1]
    other_gans = [p['gan'] for p in other_pillars]

    hiddens = month['hidden']   # 本气、中气、余气
    hidden_stars = [ten_god(day_gan, h['gan']) for h in hiddens]

    # 比劫 → 建禄 / 月劫
    if hidden_stars and hidden_stars[0] in ('比肩', '劫财'):
        name = '建禄格' if hidden_stars[0] == '比肩' else '月劫格'
        return {
            'name': name,
            'basis': f"月令本气为{hidden_stars[0]}（{hiddens[0]['gan']}）",
            'formed_hint': '以食伤、官杀、财为运用，避免比劫过旺无制',
            'broken_hint': '若比劫叠见无食伤官杀疏泄，易成破格',
            'candidates': [name],
        }

    # 其余格局：按本气 / 中气 / 余气先后找透出
    # 注：这里跳过的是比肩 / 劫财 （它们走建禄 / 月劫格，已在上方单独处理）；
    # 印、食伤、财、官杀都是正常的格局来源。
    candidates = []
    main = None
    for i, star in enumerate(hidden_stars):
        if star in ('比肩', '劫财'):
            continue
        pattern_name = _PATTERN_BY_STAR.get(star)
        if not pattern_name:
            continue
        gan_here = hiddens[i]['gan']
        tou = gan_here in other_gans
        kind = '本气' if i == 0 else ('中气' if i == 1 else '余气')
        row = {'name': pattern_name, 'star': star, 'hidden_gan': gan_here,
               'hidden_kind': kind, 'transparent': tou,
               'basis': f"月令{kind}{gan_here}为{star}" + ('，并透天干' if tou else '，未透')}
        candidates.append(row)
        if main is None and (i == 0 or tou):
            main = row
    main = main or (candidates[0] if candidates else None)

    if not main:
        return {'name': '普通格', 'basis': '月令藏干与日主关系未落入常规八格',
                'formed_hint': '', 'broken_hint': '', 'candidates': []}

    formed, broken = _pattern_judgement(main['name'], day_gan, pillars)
    return {
        'name': main['name'],
        'basis': main['basis'],
        'formed_hint': formed,
        'broken_hint': broken,
        'candidates': candidates,
    }


def _pattern_judgement(name: str, day_gan: str, pillars: list[dict]) -> tuple[str, str]:
    """给出成格 / 破格关键的常规提示。短句，不做最终裁决。"""
    stars_all: list[str] = []
    for i, p in enumerate(pillars):
        if i != 2:
            stars_all.append(ten_god(day_gan, p['gan']))
        for h in p['hidden']:
            stars_all.append(ten_god(day_gan, h['gan']))

    has = lambda s: s in stars_all

    table = {
        '正官格': (
            '喜财生官、印护官',
            '忌伤官见官、七杀混杂' + ('；本局见伤官' if has('伤官') else ''),
        ),
        '七杀格': (
            '喜食神制杀、印化杀',
            '忌财党杀而无制' + ('；本局见食神可制' if has('食神') else '；本局未见食神'),
        ),
        '正财格': (
            '喜官护财、食伤生财',
            '忌比劫夺财' + ('；本局比劫叠见' if (has('比肩') and has('劫财')) else ''),
        ),
        '偏财格': (
            '喜食伤生、官星护',
            '忌比劫夺',
        ),
        '食神格': (
            '喜财星透出成食神生财',
            '忌偏印夺食' + ('；本局见偏印' if has('偏印') else ''),
        ),
        '伤官格': (
            '喜财星泄秀、印来贵气',
            '忌伤官见官' + ('；本局见正官' if has('正官') else ''),
        ),
        '正印格': (
            '喜官星生印、不宜财破',
            '忌财星坏印' + ('；本局见正财 / 偏财' if (has('正财') or has('偏财')) else ''),
        ),
        '偏印格': (
            '喜七杀生印、食神透出化之',
            '忌偏印夺食' + ('；本局见食神' if has('食神') else ''),
        ),
    }
    return table.get(name, ('', ''))


# ── 入口 ─────────────────────────────────────────────────────────────
def analyze(chart: dict) -> dict:
    """命盘 → 事实表。供 prompt 内部接地与模型回复校验使用，不展示给用户。"""
    pillars = chart['pillars']
    day_gan = chart['day_master']['gan']
    month_zhi = pillars[1]['zhi']

    weighted = _weighted_elements(pillars)
    pct = _element_percentages(weighted)
    strength = _strength(day_gan, pillars)
    ten_gods_rank = _ten_gods_ranking(day_gan, pillars)
    branch_rel = _branch_relations(pillars)
    stem_rel = _stem_relations(pillars)
    gods = _usable_gods(day_gan, strength, pct, ten_gods_rank, month_zhi)
    pattern = _pattern(day_gan, pillars)

    return {
        'day_master': {'gan': day_gan, 'element': GAN_EL[day_gan],
                       'yang': GAN_YANG[day_gan], 'month_zhi': month_zhi},
        'elements_weighted': weighted,
        'elements_percent': pct,
        'strength': strength,
        'ten_gods': ten_gods_rank,
        'relations': {'stem': stem_rel, 'branch': branch_rel},
        'gods': gods,
        'pattern': pattern,
    }
