"""The 16 MBTI types — nickname, tagline, two-line description, and three
best-match types for the result page.

Written in plain Chinese so the result reads like a letter, not a Wikipedia
snippet. The 'match' field is the common informal pairing lore; it is not a
compatibility verdict. The page says the whole thing is for fun.
"""

TYPES: dict[str, dict] = {
    'INTJ': {
        'nick': '建筑师', 'tag': '战略家 · 独立 · 远见',
        'desc': '擅长抽象思考与长期布局,把一个复杂目标拆成清晰路径。喜欢独处时的深度思考,对低效和敷衍没有耐心。',
        'strengths': ['战略思考', '独立决策', '追求本质'],
        'blind':     ['容易显得冷淡', '对情绪不敏感', '不轻易改主意'],
        'match':     ['ENFP', 'ENTP', 'INFJ'],
    },
    'INTP': {
        'nick': '逻辑学家', 'tag': '思辨者 · 好奇 · 分析',
        'desc': '享受把一个想法翻来覆去推敲,追求理论上的优雅。对权威和规则天然怀疑,不停问为什么。',
        'strengths': ['逻辑分析', '创造性思维', '客观中立'],
        'blind':     ['拖延', '忽视感受', '沉浸在想法里不落地'],
        'match':     ['ENTJ', 'ENFJ', 'INTJ'],
    },
    'ENTJ': {
        'nick': '指挥官', 'tag': '天生领导 · 果断 · 目标感',
        'desc': '天生的组织者,擅长看清全局并带人往一个方向走。对低效、情绪化和模糊缺乏耐心。',
        'strengths': ['战略执行', '果断决策', '激励他人'],
        'blind':     ['过于强势', '不耐心倾听', '要求过高'],
        'match':     ['INTP', 'INFP', 'ENFP'],
    },
    'ENTP': {
        'nick': '辩论家', 'tag': '脑洞 · 善辩 · 挑战常规',
        'desc': '喜欢挑战主流观点,从不同角度戳破既定答案。擅长脑暴和即兴创造,对重复性事务不耐烦。',
        'strengths': ['创新思维', '快速学习', '辩论能力'],
        'blind':     ['不喜欢收尾', '容易挑衅', '承诺后分心'],
        'match':     ['INFJ', 'INTJ', 'ENFJ'],
    },
    'INFJ': {
        'nick': '提倡者', 'tag': '理想主义 · 深思 · 共情',
        'desc': '对人和世界有深刻的直觉,追求有意义的事。表面温和,内心有坚定的信念,不轻易动摇。',
        'strengths': ['深度共情', '洞察力强', '坚定信念'],
        'blind':     ['完美主义', '容易自我牺牲', '难以说不'],
        'match':     ['ENTP', 'ENFP', 'INTJ'],
    },
    'INFP': {
        'nick': '调停者', 'tag': '温柔 · 真诚 · 理想',
        'desc': '对内心世界敏感,相信每个人都值得被理解。做事更看重价值而不是效率,安静但坚韧。',
        'strengths': ['真诚待人', '创造性', '价值感强'],
        'blind':     ['逃避冲突', '容易自我怀疑', '不切实际'],
        'match':     ['ENFJ', 'ENTJ', 'INFJ'],
    },
    'ENFJ': {
        'nick': '主人公', 'tag': '鼓舞者 · 温暖 · 利他',
        'desc': '天生关心他人成长,擅长读懂情绪并带动一群人。有强烈的责任感,有时会把别人的事扛在自己肩上。',
        'strengths': ['感染力强', '组织能力', '同理心'],
        'blind':     ['过度负责', '难以拒绝', '情绪容易被牵动'],
        'match':     ['INFP', 'ISFP', 'INTP'],
    },
    'ENFP': {
        'nick': '竞选者', 'tag': '热情 · 创意 · 自由',
        'desc': '对可能性充满热情,看见新事物就想去试。擅长激发别人,但对细节和重复缺乏耐心。',
        'strengths': ['感染他人', '创意无穷', '适应变化'],
        'blind':     ['容易分心', '承诺多过交付', '情绪波动'],
        'match':     ['INTJ', 'INFJ', 'ENFJ'],
    },
    'ISTJ': {
        'nick': '物流师', 'tag': '可靠 · 务实 · 有秩序',
        'desc': '把事情做对、做完,是对自己的基本要求。重视传统、责任和承诺,对花哨的新概念保持警觉。',
        'strengths': ['可靠责任心', '组织能力', '注重细节'],
        'blind':     ['不喜欢变化', '偏保守', '对情绪不敏感'],
        'match':     ['ESFP', 'ESTP', 'ISFJ'],
    },
    'ISFJ': {
        'nick': '守卫者', 'tag': '贴心 · 细致 · 守护',
        'desc': '默默地照顾身边的人,记得每个人的喜好。做事认真细致,不喜欢被推到聚光灯下。',
        'strengths': ['贴心周到', '责任心强', '实际能干'],
        'blind':     ['忽视自己需求', '难以说不', '不喜欢冲突'],
        'match':     ['ESFP', 'ESTP', 'ISTJ'],
    },
    'ESTJ': {
        'nick': '总经理', 'tag': '执行力 · 原则 · 组织',
        'desc': '相信规则、秩序和责任。擅长把事情组织起来并推进到底,对懒散和含糊没耐心。',
        'strengths': ['执行力强', '组织清晰', '原则明确'],
        'blind':     ['不够灵活', '直来直去', '难以放手'],
        'match':     ['ISFP', 'ISTP', 'ESFJ'],
    },
    'ESFJ': {
        'nick': '执政官', 'tag': '合群 · 热心 · 守护和谐',
        'desc': '天生照顾周围人的感受,维护群体和谐。喜欢被需要,面对冲突会主动调解。',
        'strengths': ['社交能力', '关心他人', '责任感'],
        'blind':     ['过度在意评价', '难以说不', '避免冲突'],
        'match':     ['ISFP', 'ISTP', 'ESTJ'],
    },
    'ISTP': {
        'nick': '鉴赏家', 'tag': '冷静 · 动手 · 实际',
        'desc': '喜欢把东西拆开看看它怎么运作,擅长动手解决问题。独立、冷静,不喜欢被规则束缚。',
        'strengths': ['动手能力', '冷静应变', '独立'],
        'blind':     ['表达情感少', '容易孤立', '难以长期承诺'],
        'match':     ['ESFJ', 'ESTJ', 'ISFJ'],
    },
    'ISFP': {
        'nick': '探险家', 'tag': '艺术家 · 温柔 · 当下',
        'desc': '敏感、温柔,活在当下。对美和体验很敏锐,但不喜欢被规划和催促。',
        'strengths': ['审美敏感', '温柔随和', '活在当下'],
        'blind':     ['回避冲突', '容易被忽略', '难做长期计划'],
        'match':     ['ESFJ', 'ENFJ', 'ESTJ'],
    },
    'ESTP': {
        'nick': '企业家', 'tag': '行动派 · 冒险 · 现实',
        'desc': '相信"做了再说",对新机会敏感。擅长临场应变,但对长期规划缺乏耐心。',
        'strengths': ['行动迅速', '适应力强', '感染力'],
        'blind':     ['冒险不计后果', '不耐烦细节', '不太想未来'],
        'match':     ['ISTJ', 'ISFJ', 'ESFP'],
    },
    'ESFP': {
        'nick': '表演者', 'tag': '热情 · 当下 · 带动气氛',
        'desc': '活在当下,擅长调动气氛,把派对点亮。真诚、慷慨,但不喜欢被规则和远方束缚。',
        'strengths': ['感染力强', '享受当下', '待人热情'],
        'blind':     ['不想做计划', '容易冲动', '避免严肃话题'],
        'match':     ['ISTJ', 'ISFJ', 'ESTP'],
    },
}

# All 16 covered.
assert set(TYPES.keys()) == {
    a+b+c+d for a in 'EI' for b in 'SN' for c in 'TF' for d in 'JP'
}
