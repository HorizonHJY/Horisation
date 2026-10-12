"""The BaZi chart — everything that is arithmetic, and nothing that is opinion.

The model is never asked to work out a pillar. Language models get stems and
branches wrong often enough that a reading built on their arithmetic is a
reading of the wrong person. So the four pillars, ten gods, hidden stems,
growth stages, voids, na yin, luck pillars and the start of luck all come from
lunar_python (6tail, MIT) — checked against 问真八字 on a known chart — and the
model gets them as a finished table to interpret.

What lunar_python does not do is done here, in small tables:
  - true solar time (longitude + equation of time, and China's 1986–91 DST)
  - 自坐 (each stem's growth stage on its own branch)
  - a handful of common 神煞, by the most common lookup; schools differ
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

from lunar_python import Solar

GAN = '甲乙丙丁戊己庚辛壬癸'
ZHI = '子丑寅卯辰巳午未申酉戌亥'
GAN_EL = dict(zip(GAN, '木木火火土土金金水水'))
ZHI_EL = dict(zip(ZHI, '水土木木土火火土金金土水'))
ELEMENTS = ['木', '火', '土', '金', '水']

STAGES = ['长生', '沐浴', '冠带', '临官', '帝旺', '衰', '病', '死', '墓', '绝', '胎', '养']
# Where each stem's 长生 falls; yang stems run forward from it, yin backward.
_CHANG_SHENG = dict(zip(GAN, '亥午寅酉寅酉巳子申卯'))

# China observed summer time 1986–1991: clocks an hour ahead, 02:00 to 02:00.
_CN_DST = {
    1986: ((5, 4), (9, 14)), 1987: ((4, 12), (9, 13)), 1988: ((4, 17), (9, 11)),
    1989: ((4, 16), (9, 17)), 1990: ((4, 15), (9, 16)), 1991: ((4, 14), (9, 15)),
}

PILLARS = [('year', '年柱', 'Year'), ('month', '月柱', 'Month'), ('day', '日柱', 'Day'), ('time', '时柱', 'Time')]

# ── 神煞 (common lookups) ─────────────────────────────────────────────
_TIAN_YI = {'甲': '丑未', '戊': '丑未', '庚': '丑未', '乙': '子申', '己': '子申',
            '丙': '亥酉', '丁': '亥酉', '辛': '寅午', '壬': '卯巳', '癸': '卯巳'}
_WEN_CHANG = dict(zip(GAN, '巳午申酉申酉亥子寅卯'))
_LU = dict(zip(GAN, '寅卯巳午巳午申酉亥子'))
_YANG_REN = {'甲': '卯', '丙': '午', '戊': '午', '庚': '酉', '壬': '子'}
_HONG_YAN = dict(zip(GAN, '午午寅未辰辰戌酉子申'))
_GROUP = {z: g for g, zs in {'申子辰': '申子辰', '寅午戌': '寅午戌', '巳酉丑': '巳酉丑', '亥卯未': '亥卯未'}.items() for z in zs}
_TAO_HUA = {'申子辰': '酉', '寅午戌': '卯', '巳酉丑': '午', '亥卯未': '子'}
_YI_MA = {'申子辰': '寅', '寅午戌': '申', '巳酉丑': '亥', '亥卯未': '巳'}
_HUA_GAI = {'申子辰': '辰', '寅午戌': '戌', '巳酉丑': '丑', '亥卯未': '未'}


def zi_zuo(gan: str, zhi: str) -> str:
    start = ZHI.index(_CHANG_SHENG[gan])
    step = 1 if GAN.index(gan) % 2 == 0 else -1
    return STAGES[((ZHI.index(zhi) - start) * step) % 12]


def year_ganzhi(year: int) -> str:
    """The sexagenary name of a calendar year (it turns at 立春, early Feb)."""
    return GAN[(year - 4) % 10] + ZHI[(year - 4) % 12]


def _equation_of_time(d: date) -> float:
    """Minutes the sundial runs ahead of the clock (NOAA approximation)."""
    b = 2 * math.pi * (d.timetuple().tm_yday - 81) / 364
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def _in_cn_dst(clock: datetime) -> bool:
    span = _CN_DST.get(clock.year)
    if not span:
        return False
    (m1, d1), (m2, d2) = span
    return datetime(clock.year, m1, d1, 2) <= clock < datetime(clock.year, m2, d2, 2)


def true_solar_time(clock: datetime, longitude: float, utc_offset: float = 8) -> dict:
    dst = utc_offset == 8 and _in_cn_dst(clock)
    standard = clock - timedelta(hours=1) if dst else clock
    lon_min = (longitude - 15 * utc_offset) * 4
    eot_min = _equation_of_time(standard.date())
    true = standard + timedelta(minutes=lon_min + eot_min)
    return {
        'clock': clock.strftime('%Y-%m-%d %H:%M'),
        'true': true.strftime('%Y-%m-%d %H:%M'),
        'dst_applied': dst,
        'longitude_minutes': round(lon_min, 1),
        'eot_minutes': round(eot_min, 1),
        '_dt': true,
    }


def _shen_sha(pillars: list[dict]) -> None:
    """Attach the common 神煞 to each pillar, in place."""
    day_gan, year_gan = pillars[2]['gan'], pillars[0]['gan']
    year_zhi, day_zhi = pillars[0]['zhi'], pillars[2]['zhi']
    for i, p in enumerate(pillars):
        z, found = p['zhi'], []
        if z in _TIAN_YI[day_gan] or z in _TIAN_YI[year_gan]:
            found.append('天乙贵人')
        if z == _WEN_CHANG[day_gan]:
            found.append('文昌')
        if z == _LU[day_gan]:
            found.append('禄神')
        if z == _YANG_REN.get(day_gan):
            found.append('羊刃')
        if z == _HONG_YAN[day_gan]:
            found.append('红艳')
        # Branch-group stars: looked up from the year and the day branch, never
        # landing on the pillar they were looked up from.
        for name, table in (('桃花', _TAO_HUA), ('驿马', _YI_MA), ('华盖', _HUA_GAI)):
            hits = (i != 0 and z == table[_GROUP[year_zhi]]) or (i != 2 and z == table[_GROUP[day_zhi]])
            if hits:
                found.append(name)
        p['shen_sha'] = found


def _elements(pillars: list[dict]) -> dict:
    """Weighted five-element tally: a visible stem 1, a branch's hidden stems
    1 / 0.5 / 0.3 in order. A picture of balance, not a verdict."""
    score = {e: 0.0 for e in ELEMENTS}
    for p in pillars:
        score[GAN_EL[p['gan']]] += 1
        for w, h in zip((1, 0.5, 0.3), p['hidden']):
            score[h['element']] += w
    return {e: round(v, 1) for e, v in score.items()}


def build_chart(*, birth: datetime, longitude: float, utc_offset: float, male: bool,
                place: str, today: date | None = None) -> dict:
    today = today or date.today()
    st = true_solar_time(birth, longitude, utc_offset)
    t = st.pop('_dt')

    lunar = Solar.fromYmdHms(t.year, t.month, t.day, t.hour, t.minute, 0).getLunar()
    ec = lunar.getEightChar()
    ec.setSect(1)      # 23:00 starts the next day — the common default

    pillars = []
    for key, label, f in PILLARS:
        g = lambda k: getattr(ec, f'get{f}{k}')()
        gz = getattr(ec, f'get{f}')()
        gan, zhi = gz[0], gz[1]
        hidden = [{'gan': h, 'element': GAN_EL[h], 'star': s} for h, s in zip(g('HideGan'), g('ShiShenZhi'))]
        pillars.append({
            'key': key, 'label': label, 'gan': gan, 'zhi': zhi,
            'gan_element': GAN_EL[gan], 'zhi_element': ZHI_EL[zhi],
            'main_star': ('元男' if male else '元女') if key == 'day' else g('ShiShenGan'),
            'hidden': hidden,
            'di_shi': g('DiShi'),
            'zi_zuo': zi_zuo(gan, zhi),
            'xun_kong': g('XunKong'),
            'na_yin': g('NaYin'),
        })
    _shen_sha(pillars)

    yun = ec.getYun(1 if male else 0)
    da = yun.getDaYun()
    da_yun = []
    for d in da[1:10]:
        gz = d.getGanZhi()
        da_yun.append({
            'ganzhi': gz, 'start_year': d.getStartYear(), 'end_year': d.getEndYear(),
            'start_age': d.getStartAge(), 'end_age': d.getEndAge(),
            'gan_element': GAN_EL[gz[0]], 'zhi_element': ZHI_EL[gz[1]],
            'current': d.getStartYear() <= today.year <= d.getEndYear(),
        })

    return {
        'input': {'place': place, 'longitude': longitude, 'utc_offset': utc_offset,
                  'gender': 'male' if male else 'female'},
        'solar_time': st,
        'gender_label': '乾造' if male else '坤造',
        'lunar': f"{lunar.getYearInGanZhi()}年 {lunar.getMonthInChinese()}月{lunar.getDayInChinese()} {lunar.getTimeZhi()}时",
        'day_master': {'gan': pillars[2]['gan'], 'element': pillars[2]['gan_element']},
        'pillars': pillars,
        'elements': _elements(pillars),
        'yun': {
            'start': {'years': yun.getStartYear(), 'months': yun.getStartMonth(), 'days': yun.getStartDay(),
                      'date': yun.getStartSolar().toYmd()},
            'xiao_yun': {'from_age': da[0].getStartAge(), 'to_age': da[0].getEndAge(),
                         'from_year': da[0].getStartYear(), 'to_year': da[0].getEndYear()},
            'da_yun': da_yun,
        },
        'liu_nian': [{'year': y, 'ganzhi': year_ganzhi(y), 'age': y - birth.year}
                     for y in range(birth.year + 1, today.year + 1)],
        'today': today.isoformat(),
    }
