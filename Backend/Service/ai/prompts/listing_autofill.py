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

PROMPT_VERSION = 'listing-autofill-v1'

# These must match market_db default categories exactly. Order ≈ most common.
_CATEGORIES = [
    ('clothing',    '衣服'),
    ('furniture',   '家具'),
    ('kitchen',     '厨具'),
    ('electronics', '电子产品'),
    ('beauty',      '美妆'),
    ('books',       '书籍'),
    ('other',       '其他'),
]

_SYSTEM = f"""你帮用户在二手市集 Arch Bay 快速把一张商品照片转成挂单草稿。只做视觉识别。

规则:
- 看图说话,不猜品牌和型号。看不清就省略,不编。
- 标题 ≤ 20 字,先用中文,写能让买家一眼知道是什么的词;必要时括号补英文。
- 描述 ≤ 140 字,写能看到的信息:主要颜色、材质、尺寸估算、磨损迹象。不写营销话术。
- category 必须从这几个 slug 里选: {', '.join(s for s, _ in _CATEGORIES)}。看不准选 other。
- price_hint_low / price_hint_high: 对二手价格的粗估,单位美元,整数。拿不准都给 null。
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
