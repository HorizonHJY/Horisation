"""listing_autofill: vision prompt that reads one listing photo and
returns a draft of the title, description, category, and price hint.

Output is strict JSON so the New Listing form can prefill fields without
asking the seller to retype a wall of text. The schema is small on purpose:
anything the model is not sure about is left blank — a half-right title
beats a confidently-wrong one for a used-goods marketplace among friends.
"""

from __future__ import annotations

import json
import re

PROMPT_VERSION = 'listing-autofill-v2'

# These must match market_db default categories exactly. Order ≈ most common.
# v2 (2026-11-03): split out accessories / decor / sports / living after the
# existing 34 listings showed beauty was 100% tattoos and lving was a typo
# catch-all — the broader slugs below carry the real content categories.
_CATEGORIES = [
    ('clothing',    '衣服'),
    ('electronics', '电子产品'),
    ('kitchen',     '厨具'),
    ('furniture',   '家具'),
    ('decor',       '家居装饰'),
    ('accessories', '饰品'),
    ('sports',      '运动用品'),
    ('living',      '日用'),
    ('beauty',      '美妆'),
    ('books',       '书籍'),
    ('other',       '其他'),
]

_SYSTEM = f"""你帮用户在二手市集 Arch Bay 快速把一张商品照片转成挂单草稿。只做视觉识别。

输出固定格式:

**brand**: 品牌名(如 Logitech 、小米、Dyson)。看不到或不确定→ 留空字符串。不要猜。
**object**: 物品名(如 机械键盘、吐司机、耳饰)。2-6 个字最佳。
**condition**: 新旧程度。从这几个中选:全新未拆/99新/95新/9成新/8成新/7成新以下。是否拆封看包装和磨损;有磨损就是 9 成新以下。
**usage_time**: 估计使用时长(如 未使用/使用1年/使用6个月)。全新就写 未使用。假如拿不准→留空。
**function_status**: 功能状态。默认 功能正常;看到损坏(谢裂、扣子掉了、屏幕碎)→写具体现象。
**color**: 颜色描述(如 石墨蓝、梅攥射影绿)。必填。

生成规则:
- title = brand + object(有品牌时);否则仅 object。例:「罗技机械键盘」或「机械键盘」。≤ 20 字。
- description = condition + · + usage_time(有就包含) + · + function_status + · + color。用 · 连接。译成自然中文、不要写营销话。例:「95新 · 使用6个月 · 功能正常 · 石墨蓝配白色编码键」。≤ 140 字。
- category 必须从这几个 slug 里选: {', '.join(s for s, _ in _CATEGORIES)}。看不准选 other。小摆件→ decor,耳饰首饰纹身贴→ accessories,球拍麻将羽毛球→ sports,收纳盒耳机壳水杯→ living。
- price_hint_low / price_hint_high: 对二手价格粗估,美元整数。拿不准都给 null。
- 如果照片明显不是商品(人脸、风景、截图),返回 {{"error": "not_a_product"}}。

只输出 JSON,不要解释,不要代码块。schema:
{{
  "title": string,
  "description": string,
  "category": string,
  "price_hint_low": number | null,
  "price_hint_high": number | null
}}
"""


def build(image_data_url: str) -> list[dict]:
    """DeepSeek-VL2 uses OpenAI-style vision messages: content is a list of
    `{type:'text'|'image_url', ...}` blocks on the user turn.
    """
    return [
        {'role': 'system', 'content': _SYSTEM.strip()},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': '这是用户要上架的商品照片,请给出草稿。'},
            {'type': 'image_url', 'image_url': {'url': image_data_url}},
        ]},
    ]


_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def parse(text: str) -> dict | None:
    """Pull the first JSON object out of the reply and normalise it. Returns
    None if the reply isn't usable JSON — the caller still gets `text` and
    can show the raw string if it wants.
    """
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get('error') == 'not_a_product':
        return {'error': 'not_a_product'}

    valid_slugs = {s for s, _ in _CATEGORIES}
    category = str(obj.get('category') or 'other').strip()
    if category not in valid_slugs:
        category = 'other'

    def _price(v):
        try:
            n = int(float(v))
            return n if 0 < n < 100000 else None
        except (TypeError, ValueError):
            return None

    return {
        'title':           str(obj.get('title') or '').strip()[:60],
        'description':     str(obj.get('description') or '').strip()[:400],
        'category':        category,
        'price_hint_low':  _price(obj.get('price_hint_low')),
        'price_hint_high': _price(obj.get('price_hint_high')),
    }
