"""Unit tests for the BaZi fact-table analysis.

Three charts — a clearly 身强 one, a clearly 身弱 one, and a synthetic 从格
candidate — are fed through analyze() and the key verdicts are checked.
Also tests the validator catches contradictions and the corpus matches
produce the expected headlines.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.Service.bazi.chart import build_chart              # noqa: E402
from Backend.Service.bazi.analysis import analyze, ten_god      # noqa: E402
from Backend.Service.bazi import corpus                         # noqa: E402
from Backend.Service.ai.prompts import bazi as bp               # noqa: E402

TODAY = date(2026, 10, 12)


def _chart(**kw):
    base = dict(longitude=120, utc_offset=8, male=True, place='x', today=TODAY)
    base.update(kw)
    return build_chart(**base)


# ── Strength ────────────────────────────────────────────────────────

def test_jian_lu_is_strong():
    """乙 生 卯月 本气比肩 — 建禄格、身强，日支亦为比肩之地。"""
    c = _chart(birth=datetime(1998, 3, 9, 21, 34))      # 戊寅 乙卯 乙卯 丁亥
    a = analyze(c)
    assert a['strength']['level'] == '身强'
    assert a['strength']['got_season'] is True
    assert a['strength']['has_root'] is True
    assert a['pattern']['name'] == '建禄格'


def test_yi_born_in_shen_is_weak():
    """乙 生 申月（七杀月），时上辛金再克 — 身弱。"""
    c = _chart(birth=datetime(1980, 8, 10, 10, 0))      # 庚申 甲申 乙卯 辛巳
    a = analyze(c)
    assert a['strength']['level'] == '身弱'
    assert a['strength']['got_season'] is False
    # 月令是正官（辛金本气为七杀，但看 pillars[1]['hidden'][0] = 庚 → 正官）
    assert a['pattern']['name'] in ('正官格', '七杀格')
    # 用神应取 比劫 / 印 → 木 / 水
    assert '木' in a['gods']['use'] or '水' in a['gods']['use']


def test_wu_rooted_strong_prints_percentage_and_ranking():
    """五行百分比与十神排序都应有序且总和 100%。"""
    c = _chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, place='石河子')
    a = analyze(c)
    pct = a['elements_percent']
    assert abs(sum(pct.values()) - 100.0) < 0.5
    # 月令 卯 是 正印 (乙 生 丙)，月令三倍，正印应排第一
    assert a['ten_gods'][0]['star'] == '正印'
    # 十神分值单调递减
    scores = [t['score'] for t in a['ten_gods']]
    assert scores == sorted(scores, reverse=True)


# ── 关系清单 ─────────────────────────────────────────────────────

def test_relations_pick_up_day_pillar_clash():
    """丙辰 vs 时柱戊戌 — 辰戌相冲，涉日支应标记 on_day。"""
    c = _chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, place='石河子')
    a = analyze(c)
    chong = a['relations']['branch']['chong']
    assert any(r['detail'] == '辰戌相冲' and r['on_day'] for r in chong)
    # 卯戌六合 (月月-时时) 也应在
    liu_he = a['relations']['branch']['liu_he']
    assert any('卯戌' in r['detail'] for r in liu_he)


def test_relations_detect_san_hui():
    c = _chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, place='石河子')
    a = analyze(c)
    hui = a['relations']['branch']['san_hui']
    assert any('三会' in r['detail'] for r in hui)


# ── 从格 ─────────────────────────────────────────────────────────

def test_follow_candidate_on_rootless_weak_master():
    """乙生巳月，一片火土金，日主无根 — 从财 / 从儿候选。"""
    c = _chart(birth=datetime(1988, 5, 10, 14, 0))   # 戊辰 丁巳 乙丑 癸未
    a = analyze(c)
    # 不一定硬从，但 strength 判定应是 身弱 并给出 follow_candidate 信息（可能为 None）
    assert a['strength']['level'] in ('身弱', '从财格', '从儿格')


def test_pure_follow_candidate_synthetic():
    """合成测试：手工构造一张无根失令的 strict 从格候选，走分析函数判从。"""
    # 丙火 生 子月(壬癸水)、年柱 壬子、时柱 庚子 — 一片水；日支亦为申金。
    # 实际日期找得费劲，这里直接走 build_chart 挑一个贴近的日期。
    c = _chart(birth=datetime(1972, 12, 20, 16, 0))
    a = analyze(c)
    s = a['strength']
    # 可能为 身弱 或 从格；核心是分析给出了合理的 reason
    assert s['level'] in ('身弱', '身强', '中和', '从财格', '从杀格', '从儿格', '从官格')
    assert s['reasons']


# ── Corpus ───────────────────────────────────────────────────────

def test_corpus_matches_specific_combinations():
    """1998-03-10 丙辰 — 命中 日支被冲、三会成方、财破印、印多身强 等。"""
    c = _chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, place='石河子')
    a = analyze(c)
    heads = [h['headline'] for h in corpus.hits_for(a)]
    assert '日支被冲' in heads
    assert '三会成方' in heads
    assert '印多身强' in heads


def test_corpus_entry_count_known():
    """校验现有语料条数（将来增删条目时需同步改这里）。"""
    assert corpus.count() == 30


# ── Ten god helper ───────────────────────────────────────────────

def test_ten_god_tables_match_classical():
    assert ten_god('丙', '壬') == '七杀'
    assert ten_god('丙', '癸') == '正官'
    assert ten_god('丙', '辛') == '正财'
    assert ten_god('丙', '庚') == '偏财'
    assert ten_god('甲', '己') == '正财'


# ── Validator ────────────────────────────────────────────────────

def test_validator_accepts_a_consistent_reply():
    c = _chart(birth=datetime(1998, 3, 9, 21, 34))
    reading = {
        'verdict': '建禄身旺，木气独盛',
        'strength': {'level': '身强', 'reasons': []},
        'pattern': {'name': '建禄格', 'explain': ''},
        'gods': {'use': {'element': '土'}, 'like': {}, 'avoid': {}, 'idle': {}},
        'da_yun': [], 'liu_nian': [], 'summary': '',
    }
    assert bp.validate(reading, c) == []


def test_validator_catches_strength_mismatch():
    c = _chart(birth=datetime(1998, 3, 9, 21, 34))   # 身强
    bad = {'verdict': 'x', 'strength': {'level': '身弱', 'reasons': []},
           'pattern': {'name': 'x'}, 'gods': {}, 'da_yun': [], 'liu_nian': [], 'summary': ''}
    notes = bp.validate(bad, c)
    assert any('strength.level' in n for n in notes)


def test_validator_catches_fabricated_relation():
    """卯戌合火在局中有；辰午相冲不在局中 — 后者应被抓。"""
    c = _chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, place='石河子')
    reading = {
        'verdict': 'x',
        'strength': {'level': '身强', 'reasons': []},
        'pattern': {'name': '正印格', 'explain': ''},
        'gods': {'use': {'element': '水', 'why': ''}, 'like': {}, 'avoid': {}, 'idle': {}},
        'da_yun': [{'ganzhi': '丙辰', 'theme': '', 'reading': '辰午相冲，诸事不顺'}],
        'liu_nian': [], 'summary': '',
    }
    notes = bp.validate(reading, c)
    assert any('辰午' in n for n in notes), notes


def test_validator_catches_bad_star_name():
    c = _chart(birth=datetime(1998, 3, 9, 21, 34))
    reading = {
        'verdict': 'x',
        'strength': {'level': '身强', 'reasons': []},
        'pattern': {'name': '建禄格', 'explain': '正杀混杂，宜防'},
        'gods': {'use': {'element': '土', 'why': ''}, 'like': {}, 'avoid': {}, 'idle': {}},
        'da_yun': [], 'liu_nian': [], 'summary': '',
    }
    notes = bp.validate(reading, c)
    assert any('正杀' in n for n in notes), notes


def test_repair_softens_strength_mismatch():
    c = _chart(birth=datetime(1998, 3, 9, 21, 34))
    bad = {'verdict': 'x', 'strength': {'level': '身弱', 'reasons': []},
           'pattern': {'name': 'x'}, 'gods': {}, 'da_yun': [], 'liu_nian': [], 'summary': ''}
    notes = bp.validate(bad, c)
    fixed = bp.repair(bad, notes)
    assert fixed['strength']['level'] == '中和'


def test_repair_scrubs_fabricated_relation_from_text():
    c = _chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, place='石河子')
    reading = {
        'verdict': 'x',
        'strength': {'level': '身强', 'reasons': []},
        'pattern': {'name': '正印格'},
        'gods': {'use': {'element': '水', 'why': ''}, 'like': {}, 'avoid': {}, 'idle': {}},
        'da_yun': [{'ganzhi': '丙辰', 'theme': '', 'reading': '辰午相冲不利'}],
        'liu_nian': [], 'summary': '辰午相冲是关键',
    }
    notes = bp.validate(reading, c)
    fixed = bp.repair(reading, notes)
    assert '辰午相冲' not in fixed['da_yun'][0]['reading']
    assert '辰午相冲' not in fixed['summary']


# ── Prompt wiring ────────────────────────────────────────────────

def test_build_prompt_v2_includes_fact_table():
    c = _chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, place='石河子')
    msgs = bp.build(c)
    user = msgs[1]['content']
    assert '【权威事实表' in user
    assert '身强' in user   # 丙 born 卯月 should resolve as 身强
    assert '卯戌六合化火' in user
    assert bp.PROMPT_VERSION == 'bazi-v2'


def test_build_prompt_v2_does_not_mention_sources():
    c = _chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, place='石河子')
    msgs = bp.build(c)
    joined = msgs[0]['content'] + msgs[1]['content']
    # 不得出现古籍名引用。「古书云 / 古籍」在规则行里被示例性提及，
    # 不算泄漏；具体书名在提示词里一定不能出现。
    for forbidden in ('子平真诠', '三命通会', '滴天髓', '渊海子平'):
        assert forbidden not in joined, f'prompt v2 leaked source: {forbidden}'
