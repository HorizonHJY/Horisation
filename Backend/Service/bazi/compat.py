"""The BaZi 合盘 — arithmetic for two charts together.

The reading is the AI's; the arithmetic — day-pillar合冲、生肖、五行互补、十神
关系 — is here, so the model never has to count stems and branches again.

Left side is 本命 (self), right side is 对象 (other). The structure is small,
and shaped so the page can lay it out as cards and bars without doing work.
"""

from __future__ import annotations

from .chart import GAN, ZHI, GAN_EL, ZHI_EL, ELEMENTS

# ── 关系表 ─────────────────────────────────────────────────────────
# 天干五合：阳合阴 → 化五行
GAN_HE = {
    frozenset('甲己'): '土', frozenset('乙庚'): '金',
    frozenset('丙辛'): '水', frozenset('丁壬'): '木',
    frozenset('戊癸'): '火',
}
# 天干相克 (克者→被克者)
GAN_KE = {('甲', '戊'), ('乙', '己'), ('丙', '庚'), ('丁', '辛'), ('戊', '壬'),
          ('己', '癸'), ('庚', '甲'), ('辛', '乙'), ('壬', '丙'), ('癸', '丁')}

# 地支六合
ZHI_LIU_HE = {
    frozenset('子丑'): '土', frozenset('寅亥'): '木', frozenset('卯戌'): '火',
    frozenset('辰酉'): '金', frozenset('巳申'): '水', frozenset('午未'): '土',
}
# 地支三合
ZHI_SAN_HE = {
    frozenset('申子辰'): '水', frozenset('亥卯未'): '木',
    frozenset('寅午戌'): '火', frozenset('巳酉丑'): '金',
}
# 地支六冲
ZHI_CHONG = [frozenset(p) for p in ['子午', '丑未', '寅申', '卯酉', '辰戌', '巳亥']]
# 地支相害
ZHI_HAI = [frozenset(p) for p in ['子未', '丑午', '寅巳', '卯辰', '申亥', '酉戌']]
# 地支相刑（无礼之刑、恃势之刑、无恩之刑、自刑略过）
ZHI_XING = [frozenset(p) for p in ['子卯', '寅巳', '巳申', '丑戌', '戌未', '丑未']]

ZHI_ANIMAL = dict(zip(ZHI, '鼠牛虎兔龙蛇马羊猴鸡狗猪'))

# 十神对照表：以 day_master 为我，对比对方干支的十神
# 阴阳同 → 比劫/食伤/财/官/印 的「正」
# 阴阳异 → 「偏」
GAN_YIN_YANG = {g: (i % 2 == 0) for i, g in enumerate(GAN)}   # True=阳


def _ten_god(me: str, other: str) -> str:
    """我为 me，看 other 对我的十神。"""
    rel = _rel(GAN_EL[me], GAN_EL[other])
    same_yy = GAN_YIN_YANG[me] == GAN_YIN_YANG[other]
    table = {
        'same':  ('比肩', '劫财'),      # 同类
        'i_gen': ('食神', '伤官'),      # 我生者
        'i_ke':  ('偏财', '正财'),      # 我克者
        'ke_i':  ('七杀', '正官'),      # 克我者
        'gen_i': ('偏印', '正印'),      # 生我者
    }
    primary, secondary = table[rel]
    return primary if same_yy else secondary


_SHENG = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
_KE   = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}


def _rel(me_el: str, other_el: str) -> str:
    if me_el == other_el: return 'same'
    if _SHENG[me_el] == other_el: return 'i_gen'
    if _KE[me_el] == other_el: return 'i_ke'
    if _KE[other_el] == me_el: return 'ke_i'
    return 'gen_i'  # 对方生我


# ── 配对判断 ──────────────────────────────────────────────────────

def _day_pair(a: dict, b: dict) -> dict:
    """日柱配对：相合/相冲/相克。日柱是夫妻宫，权重最高。"""
    ag, az = a['pillars'][2]['gan'], a['pillars'][2]['zhi']
    bg, bz = b['pillars'][2]['gan'], b['pillars'][2]['zhi']

    gan_rel = []
    gan_he = GAN_HE.get(frozenset((ag, bg)))
    if gan_he and ag != bg:
        gan_rel.append({'kind': 'he', 'detail': f'{ag}{bg}合化{gan_he}', 'element': gan_he})
    if (ag, bg) in GAN_KE: gan_rel.append({'kind': 'ke', 'detail': f'{ag}克{bg}'})
    if (bg, ag) in GAN_KE: gan_rel.append({'kind': 'ke', 'detail': f'{bg}克{ag}'})

    zhi_rel = []
    zhi_he = ZHI_LIU_HE.get(frozenset((az, bz)))
    if zhi_he and az != bz:
        zhi_rel.append({'kind': 'liu_he', 'detail': f'{az}{bz}六合化{zhi_he}', 'element': zhi_he})
    if frozenset((az, bz)) in ZHI_CHONG:
        zhi_rel.append({'kind': 'chong', 'detail': f'{az}{bz}相冲'})
    if frozenset((az, bz)) in ZHI_HAI:
        zhi_rel.append({'kind': 'hai', 'detail': f'{az}{bz}相害'})
    if frozenset((az, bz)) in ZHI_XING:
        zhi_rel.append({'kind': 'xing', 'detail': f'{az}{bz}相刑'})

    # 定级：天合地合最吉；天克地冲最凶；其余中性/偏吉偏凶
    if gan_he and zhi_he: level = 'auspicious_pair'   # 天合地合
    elif any(r['kind'] == 'ke' for r in gan_rel) and any(r['kind'] == 'chong' for r in zhi_rel):
        level = 'adverse_pair'                        # 天克地冲
    elif gan_he or zhi_he: level = 'favorable'
    elif any(r['kind'] in ('ke',) for r in gan_rel) or any(r['kind'] in ('chong', 'xing', 'hai') for r in zhi_rel):
        level = 'challenging'
    else: level = 'neutral'

    return {
        'self':   {'gan': ag, 'zhi': az, 'label': ag + az},
        'other':  {'gan': bg, 'zhi': bz, 'label': bg + bz},
        'gan_rel': gan_rel, 'zhi_rel': zhi_rel,
        'level':  level,
    }


def _zodiac(a: dict, b: dict) -> dict:
    az, bz = a['pillars'][0]['zhi'], b['pillars'][0]['zhi']
    rel = []
    if frozenset((az, bz)) in [frozenset(p) for p in ZHI_LIU_HE]:
        rel.append({'kind': 'liu_he', 'detail': f'{az}{bz}六合'})
    for group in ZHI_SAN_HE:
        if az in group and bz in group and az != bz:
            rel.append({'kind': 'san_he', 'detail': f'{az}{bz}同在三合({"".join(group)})'})
    if frozenset((az, bz)) in ZHI_CHONG:
        rel.append({'kind': 'chong', 'detail': f'{az}{bz}生肖相冲'})
    if frozenset((az, bz)) in ZHI_HAI:
        rel.append({'kind': 'hai', 'detail': f'{az}{bz}生肖相害'})
    return {
        'self':  {'zhi': az, 'animal': ZHI_ANIMAL[az]},
        'other': {'zhi': bz, 'animal': ZHI_ANIMAL[bz]},
        'rel': rel,
    }


def _elements_match(a: dict, b: dict) -> dict:
    """五行互补：谁强谁弱，哪个元素是对方的补益。"""
    aa = a['elements']; bb = b['elements']
    # 对方较强而我较弱的元素，视为对方对我的补益；反之亦然
    gift_to_self = {}
    gift_to_other = {}
    for e in ELEMENTS:
        if bb[e] - aa[e] >= 1.0:
            gift_to_self[e] = round(bb[e] - aa[e], 1)
        if aa[e] - bb[e] >= 1.0:
            gift_to_other[e] = round(aa[e] - bb[e], 1)
    return {
        'self':  {e: aa[e] for e in ELEMENTS},
        'other': {e: bb[e] for e in ELEMENTS},
        'gift_to_self':  gift_to_self,     # 对方多出来、能补我的
        'gift_to_other': gift_to_other,
    }


def _ten_gods(a: dict, b: dict) -> dict:
    """以本命日主为我，看对方四柱天干对我的十神关系。
    夫妻之间，对方的日干对我的十神，是最被看重的那一条。"""
    me = a['pillars'][2]['gan']
    cells = []
    for p in b['pillars']:
        cells.append({
            'pillar': p['label'],
            'gan': p['gan'],
            'ten_god': _ten_god(me, p['gan']),
        })
    # 对方日干对我的十神（最被看重的一条）
    key = _ten_god(me, b['pillars'][2]['gan'])
    return {'self_me': me, 'self_me_label': f"{me}{GAN_EL[me]}", 'cells': cells, 'spouse_god': key}


def _gender_pair(a: dict, b: dict) -> str:
    """乾坤 / 乾乾 / 坤坤 → 配对类型；本程序不限制必须异性，友情盘也能用。"""
    ga, gb = a['input']['gender'], b['input']['gender']
    if ga != gb: return 'romantic'            # 一乾一坤
    return 'same_gender_friendship'           # 同性，友情向


def _score(day: dict, zodiac: dict, elements: dict) -> int:
    """粗分 0-100，只是给前端一个显示用的数字，别当真。"""
    s = 60
    if day['level'] == 'auspicious_pair': s += 25
    elif day['level'] == 'favorable':     s += 12
    elif day['level'] == 'challenging':   s -= 10
    elif day['level'] == 'adverse_pair':  s -= 20
    for r in zodiac['rel']:
        if r['kind'] in ('liu_he', 'san_he'): s += 5
        elif r['kind'] == 'chong':            s -= 8
        elif r['kind'] == 'hai':              s -= 3
    s += min(10, len(elements['gift_to_self']) * 3)
    s += min(10, len(elements['gift_to_other']) * 3)
    return max(0, min(100, s))


def build_compat(chart_a: dict, chart_b: dict) -> dict:
    """两张命盘的合盘分析（纯算术，不含 AI 解读）。"""
    day = _day_pair(chart_a, chart_b)
    zodiac = _zodiac(chart_a, chart_b)
    elements = _elements_match(chart_a, chart_b)
    ten_gods = _ten_gods(chart_a, chart_b)
    gender_pair = _gender_pair(chart_a, chart_b)
    return {
        'gender_pair': gender_pair,
        'day_pair':   day,
        'zodiac':     zodiac,
        'elements':   elements,
        'ten_gods':   ten_gods,
        'score':      _score(day, zodiac, elements),
    }
