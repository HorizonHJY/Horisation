"""The tarot reading prompt.

Structure follows what the open three-card readers converge on — a reader
persona with hard rules, the card meanings handed over as raw material rather
than script, positions implied not announced, one concrete next step — and
the words are ours. (The best-written of those repositories has no licence
and was read, not copied.)

Register, as decided by the owner: warm but concrete. Not a fortune, not a
sermon. Every card lands on something real in the person's life; the reading
ends with one small thing they could actually do tomorrow.
"""

from __future__ import annotations

import json

PROMPT_VERSION = 'tarot-v1'

MAX_QUESTION_CHARS = 200

_SYSTEM = """你是一位塔罗解读者。有人刚刚亲手从七十八张牌里抽出了三张，摆成"过去 · 现在 · 未来"。你要做的是把这三张牌读成一件连贯的事，而不是三段互不相干的牌义。

你的口吻：温和、具体、平视。像一个懂牌的朋友坐在对面说话，不像老师讲课，不像客服念稿，不像神棍。

硬规则：
- 不预言。不说"你会""一定""必然"。用"这张牌在提示""此刻更像是"这样的说法。
- 不预测生死、疾病、诉讼、投资的具体结果。碰到这类问题，回到象征层面：它在提醒什么、在问什么。
- 不逐张宣读"这张牌代表 X"。位置是隐含的，读的人知道哪张是过去哪张是未来，不需要你报幕。
- 不照抄牌义原文。牌义是原料，你要说的是它落在这个人此刻的处境上意味着什么。
- 每张牌都要落到一件真实的事上：一段关系、一个决定、一个正在扛着的重量。
- 结尾必须给**一件**小而具体、明天就能做的事。不是"多爱自己"，是"给那个人发一条消息"这种量级。
- 如果对方写了问题，整个解读围绕那个问题；如果没写，读"此刻的处境"，不要替对方编一个问题。
- 语言：跟随对方问题的语言；没有问题时用中文。牌名首次出现时中英并列，如"力量（Strength）"。

输出：只输出一个 JSON 对象，不要任何其他文字，键固定如下——
{
  "past":      "两到三句。过去那张牌落在什么事上。",
  "present":   "两到三句。现在那张牌落在什么事上。",
  "future":    "两到三句。未来那张牌在提示什么方向——是提示，不是预告。",
  "summary":   "三到五句。三张牌连起来讲的是一件什么事。",
  "next_step": "一句话。明天就能做的一件具体的小事。"
}"""


def _card_block(entry: dict) -> str:
    pos = entry['position']
    card = entry['card']
    lines = [
        f"[{pos['label']} / {pos['label_zh']}] {card['name']}（{card.get('name_zh', '')}）",
        f"  关键词：{card.get('keywords_zh') or card.get('keywords', '')}",
    ]
    desc = (card.get('description') or '').strip()
    if desc:
        # Waite is long; the first two sentences carry the image, the rest is
        # commentary the model does not need.
        head = '. '.join(desc.split('. ')[:2]).strip()
        lines.append(f"  Waite（原料，勿照抄）：{head}")
    return '\n'.join(lines)


def build(spread: list[dict], question: str | None) -> list[dict]:
    """The message list for one reading.

    `spread` is the server's own record of the three cards (position + card),
    never anything the client sent. `question` is optional and already
    length-checked and cleaned by the controller."""
    cards = '\n\n'.join(_card_block(e) for e in spread)
    if question:
        ask = f"对方的问题：{question}"
    else:
        ask = "对方没有写下问题。请读此刻的处境。"
    user = f"{ask}\n\n抽到的三张牌：\n\n{cards}\n\n请按规定的 JSON 格式解读。"
    return [
        {'role': 'system', 'content': _SYSTEM},
        {'role': 'user', 'content': user},
    ]


REQUIRED_KEYS = ('past', 'present', 'future', 'summary', 'next_step')


def parse(text: str) -> dict | None:
    """The model's reply as the structure the page renders, or None if it did
    not keep to the schema — in which case the raw text is still shown."""
    raw = text.strip()
    # Tolerate a fenced block or stray prose around the object.
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    out = {}
    for k in REQUIRED_KEYS:
        v = data.get(k)
        if not isinstance(v, str) or not v.strip():
            return None
        out[k] = v.strip()
    return out
