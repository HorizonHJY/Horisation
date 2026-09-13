// Adds `name_zh` and `keywords_zh` to Backend/data/tarot_deck.json.
//
// No open Chinese tarot dataset exists (the MIT ones are English-only and the
// Chinese meaning sites are copyrighted), so this layer is authored here:
//   name_zh      — the standard Chinese names, which are fixed convention
//   keywords_zh  — short conventional upright keywords, written for this deck,
//                  NOT a translation of Waite's 1911 text
// Idempotent: run again to overwrite both fields from this file.
//
//   node scripts/tarot_add_zh.mjs

import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const DECK = join(here, '..', 'Backend', 'data', 'tarot_deck.json')

const MAJOR = {
  m00: ['愚者',     '新的开始、冒险、天真、跳出常规'],
  m01: ['魔术师',   '意志、创造力、把想法变成现实'],
  m02: ['女祭司',   '直觉、潜意识、静观其变、保守秘密'],
  m03: ['女皇',     '丰饶、滋养、感官之美、母性'],
  m04: ['皇帝',     '秩序、权威、稳定的结构、父性'],
  m05: ['教皇',     '传统、信仰、师承、遵循规范'],
  m06: ['恋人',     '爱、结合、价值观的选择、和谐'],
  m07: ['战车',     '意志力、掌控、向前推进、胜利'],
  m08: ['力量',     '柔中带刚、耐心、驯服内心的野性'],
  m09: ['隐士',     '独处、内省、寻求智慧、指引'],
  m10: ['命运之轮', '转机、周期、命运的转折、好运'],
  m11: ['正义',     '公正、因果、真相、权衡与决断'],
  m12: ['倒吊人',   '暂停、放下、换个角度、自愿的牺牲'],
  m13: ['死神',     '结束与转化、告别旧事物、新生'],
  m14: ['节制',     '平衡、调和、耐心、中庸之道'],
  m15: ['恶魔',     '束缚、欲望、执念、看清枷锁'],
  m16: ['高塔',     '骤变、崩塌、真相揭露、打破幻象'],
  m17: ['星星',     '希望、疗愈、灵感、宁静的信心'],
  m18: ['月亮',     '不安、幻象、潜意识、走过迷雾'],
  m19: ['太阳',     '喜悦、成功、活力、光明与坦诚'],
  m20: ['审判',     '觉醒、召唤、清算过去、重生'],
  m21: ['世界',     '圆满、完成、整合、周期的达成'],
}

const SUIT = {
  Wands:     { zh: '权杖', theme: '行动与热情' },
  Cups:      { zh: '圣杯', theme: '情感与关系' },
  Swords:    { zh: '宝剑', theme: '思维与冲突' },
  Pentacles: { zh: '星币', theme: '物质与工作' },
}
const RANK = {
  1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七',
  8: '八', 9: '九', 10: '十', 11: '侍从', 12: '骑士', 13: '王后', 14: '国王',
}
const rankName = (suitZh, n) => (n <= 10 ? `${suitZh}${RANK[n]}` : `${suitZh}${RANK[n]}`)

// Upright keywords, minor arcana. Indexed by suit then rank 1–14.
const MINOR = {
  Wands: {
    1: '灵感、新的动力、创造的火种',
    2: '规划、展望、下一步的选择',
    3: '远见、扩张、等待成果',
    4: '庆祝、稳固的基础、归属',
    5: '竞争、摩擦、各持己见',
    6: '胜利、认可、公开的成功',
    7: '坚守、捍卫立场、迎战挑战',
    8: '迅速推进、消息到来、动能',
    9: '坚持、警惕、最后一段路',
    10: '重担、责任过载、快到终点',
    11: '热情的消息、探索、跃跃欲试',
    12: '冲劲、冒险、行动优先',
    13: '自信、温暖、有感染力',
    14: '远见的领导、魄力、掌控全局',
  },
  Cups: {
    1: '新的感情、情感的丰盈、爱',
    2: '结合、相互吸引、伙伴关系',
    3: '友谊、庆祝、共同的喜悦',
    4: '倦怠、错过眼前的机会、自省',
    5: '失落、哀悼、留意仍在的东西',
    6: '怀旧、纯真、旧人旧事重现',
    7: '幻想、选择过多、看清真实',
    8: '离开、放下、寻找更深的意义',
    9: '满足、愿望达成、自得其乐',
    10: '圆满的感情、家庭和睦、幸福',
    11: '感性的消息、创意、真诚的心意',
    12: '浪漫、追求理想、有情人',
    13: '同理心、直觉、温柔的关怀',
    14: '情绪成熟、包容、平静的智慧',
  },
  Swords: {
    1: '清晰、真相、新的想法、突破',
    2: '僵持、回避决定、需要看清',
    3: '心碎、伤痛、必要的释放',
    4: '休整、恢复、暂时退后',
    5: '冲突、代价高昂的胜利、放手',
    6: '过渡、离开困境、渐渐平静',
    7: '策略、隐瞒、另辟蹊径',
    8: '受困、自我设限、其实有出路',
    9: '焦虑、失眠、放大的担忧',
    10: '谷底、结束、最坏已过',
    11: '好奇、直言、新的观点',
    12: '果断、迅速、直冲目标',
    13: '清醒、独立、界限分明',
    14: '理性、权威、公正的判断',
  },
  Pentacles: {
    1: '新的机会、物质基础、可靠的开始',
    2: '平衡、兼顾多方、灵活应对',
    3: '协作、手艺、被认可的努力',
    4: '守成、安全感、握得太紧',
    5: '匮乏、艰难时期、寻求支持',
    6: '慷慨、给予与接受、公平',
    7: '耐心、评估进展、等待收成',
    8: '专注、精进、踏实的努力',
    9: '自足、独立、享受成果',
    10: '传承、长久的富足、家业',
    11: '学习、务实的机会、新技能',
    12: '稳步、可靠、按部就班',
    13: '务实的关怀、富足、经营有方',
    14: '成就、稳固、成功的经营',
  },
}

const deck = JSON.parse(readFileSync(DECK, 'utf8'))
let n = 0
for (const card of deck) {
  if (card.arcana === 'major') {
    const [name, kw] = MAJOR[card.id] || []
    if (!name) throw new Error(`no zh for ${card.id}`)
    card.name_zh = name
    card.keywords_zh = kw
  } else {
    const suit = SUIT[card.suit]
    const rank = Number(card.number)
    if (!suit || !RANK[rank]) throw new Error(`no zh for ${card.id} ${card.suit} ${card.number}`)
    card.name_zh = rankName(suit.zh, rank)
    card.keywords_zh = MINOR[card.suit][rank]
  }
  n++
}
writeFileSync(DECK, JSON.stringify(deck, null, 2) + '\n', 'utf8')
console.log(`wrote name_zh + keywords_zh for ${n} cards`)
console.log('sample:', deck.filter(c => ['m00', 'm08', 'c13', 'w01', 'p14'].includes(c.id)).map(c => `${c.name} = ${c.name_zh} — ${c.keywords_zh}`).join('\n        '))
