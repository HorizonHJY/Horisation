# AI Service Layer — 设计方案（待评审）

状态：**草案 / 未实现**
作者：小思
日期：2026-09-20

> 本文是设计提案，不是已落地的实现。目标是让 Horizon 决定「要不要做、做成什么样」，
> 再决定是否动手。所有代码路径均为建议，尚未写入仓库。

---

## 1. 目标

Horisation 未来的 AI 功能会不止一个（塔罗解读是第一个）。本方案要解决的
不是「塔罗怎么读牌」，而是：

- **一份 API key、一套调用逻辑、一处成本可见**，供全站所有 AI 功能共用。
- 每个新 AI 功能 = 加一个业务方法 + 一个 prompt，**不碰基础设施**。
- 额度、限流、日志、模型选择、失败降级都在这一层统一，不再各写各的。
- 现在不额外部署进程；但接口抽象干净到**将来能整体拆成独立服务**。

**非目标（本期不做）：**

- 独立部署的 AI 微服务 / 容器（等有多个后端或需要独立扩缩容再说）。
- 多 provider 路由（DeepSeek + Claude + 本地模型）。接口预留，实现先只接一个。
- 向量检索 / RAG（牌义目前是静态小数据，直接塞进 prompt 即可）。

---

## 2. 总体形态

**不做独立 App，先在 Horisation 后端内部抽一个模块。**

```
Backend/
├── Service/
│   ├── ai/
│   │   ├── __init__.py        # 暴露单例 ai
│   │   ├── client.py          # DeepSeek HTTP 客户端（唯一碰网络的地方）
│   │   ├── service.py         # AIService：quota → 调用 → 记账 → 降级
│   │   ├── quota.py           # 配额策略（按 feature/user/day）
│   │   ├── usage.py           # 用量与成本落库
│   │   ├── prompts.py         # 所有 prompt 集中放，便于调性统一
│   │   └── features/
│   │       ├── tarot.py       # interpret_tarot(...)
│   │       └── __init__.py    # 注册表：feature 名 → 处理函数
```

调用方（controller）只看到一层薄 API：

```python
from Backend.Service.ai import ai

result = ai.run(
    feature='tarot',
    user=current_username,
    payload={'question': q, 'spread': spread},
)
# result.ok / result.text / result.usage / result.error
```

**关键点：功能代码不直接 import `client.py`。** 谁都不许绕过 service 直接调网络，
否则限流和记账就漏了。

---

## 3. 核心接口契约

### 3.1 统一返回结构

所有 AI 功能，无论内部用哪个模型，返回同一个形状：

```python
@dataclass
class AIResult:
    ok: bool
    text: str | None            # 成功时的正文（markdown 允许）
    data: dict | None           # 结构化输出时（如塔罗分成 past/present/future）
    error: str | None           # 失败原因（给用户看的短句，不泄内部细节）
    usage: AIUsage              # tokens / model / cost / latency_ms
    quota: QuotaState           # used / limit / remaining / reset_at
```

前端只认这一个结构，不再解析各家 API 的奇怪返回。

### 3.2 配额标识

限流规则**不写死在 AI 层**。调用方传一个 `quota_key`，AI 层只负责计数：

```
user:zwy:tarot:2026-09-20
```

- 塔罗：一天一次 → key 含日期，天然「跨天自动重置」。
- 将来不限次的功能：传 `None`，跳过额度检查，仅记账。
- 高等级用户（vip/svip/horizon）配额不同 → 由 `quota.py` 按 role 查上限表。

### 3.3 调用者身份

每次调用必须带 `feature` 名（`'tarot'` / 以后的 `'listing-helper'` 等）。
用途：日志分类、成本按功能拆分、按功能设独立预算。**没有 feature 名直接报错**，
防止有人图省事不标。

---

## 4. 数据模型

新增两张表（`_data/market.db`）。**只增不改**，不动现有表。

### 4.1 `ai_usage` — 记账与限流的唯一事实来源

```sql
CREATE TABLE ai_usage (
  id           TEXT PRIMARY KEY,          -- uuid4
  feature      TEXT NOT NULL,             -- 'tarot'
  username     TEXT NOT NULL,
  quota_key    TEXT,                      -- 'user:zwy:tarot:2026-09-20'，可为 NULL
  model        TEXT NOT NULL,             -- 'deepseek-chat'
  prompt_tokens      INTEGER DEFAULT 0,
  completion_tokens  INTEGER DEFAULT 0,
  cost_usd     REAL DEFAULT 0,            -- 按当次单价算出的估算成本
  latency_ms   INTEGER,
  ok           INTEGER NOT NULL,          -- 1/0
  error_kind   TEXT,                      -- 'timeout' / 'rate_limit' / 'provider' / NULL
  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_ai_usage_quota ON ai_usage(quota_key, ok);
CREATE INDEX idx_ai_usage_feature_day ON ai_usage(feature, created_at);
```

**一个必须先想清楚的设计取舍：失败要不要记账？**

我建议 **成功和失败都写 `ai_usage`，但限流只数 `ok=1` 的行。**

- 失败（超时 / provider 报错）**不消耗**用户当天额度 → 用户能重试。这是对的，
  不能让一次网络抖动把用户名额吃掉。
- 但失败**要留下记录** → 否则你永远不知道 AI 到底失败率多高、钱有没有白烧。

所以限流查询是：
```sql
SELECT COUNT(*) FROM ai_usage
WHERE quota_key = ? AND ok = 1;
```

### 4.2 `tarot_readings` — 业务内容（可选，但建议有）

```sql
CREATE TABLE tarot_readings (
  id          TEXT PRIMARY KEY,
  username    TEXT NOT NULL,
  question    TEXT,
  spread_json TEXT NOT NULL,       -- 抽到的三张牌 + 位置
  reading_json TEXT,               -- AI 返回的结构化解读
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_tarot_user_day ON tarot_readings(username, created_at);
```

为什么要单独一张：`ai_usage` 是「基础设施记账」，`tarot_readings` 是「业务数据」。
塔罗以后要「回看历史牌阵」、做分享页，靠这张表。分开存，职责干净。

---

## 5. 时区 —— 必须现在定死

**这是最容易上线后才炸的坑。**

`date('now')` 是 UTC。用户在美国，本地晚上 8 点抽牌，UTC 已是次日 —— 额度会在
本地时间下午/傍晚被错误重置。两种做法：

- **方案 i（推荐）：`quota_key` 里直接写业务时区日期。**
  Python 端用 `zoneinfo` 算 America/Chicago 的日期，拼进 key。
  数据库不参与日期运算，逻辑全在一处，改时区只改一个常量。
- 方案 ii：SQL 里用 `date(created_at, '-5 hours')`。硬编码偏移，遇到夏令时会错，**不推荐**。

结论：用 `zoneinfo`，时区常量放 `ai/service.py` 顶部。

---

## 6. 调用流程（以塔罗为例）

```
POST /api/tarot/reading     body: { question }
  1. 登录校验（复用 @login_required）
  2. 校验 question：非空、长度 ≤ 200 字、去控制字符
  3. quota_key = f"user:{u}:tarot:{local_date}"
  4. quota.check(quota_key)  →  超了直接 429 + 友好提示，不抽牌不花钱
  5. 服务端抽三张牌（复用现有 draw 的 secrets 逻辑，只抽一次）
  6. 拼 prompt：问题 + 三张牌的 keywords_zh / description（不喂整副牌）
  7. client.chat(...)  →  超时/失败：写 ai_usage(ok=0)，返回可重试错误，不扣额度
  8. 成功：解析 → 结构化 reading →
       写 tarot_readings + ai_usage(ok=1)  →  返回 { spread, reading, quota }
```

**为什么抽牌和解读必须在同一个后端流程里：**
现有 `tarot_controller.py` 注释已经写明「能在 devtools 重抽的牌阵不算牌阵」。
同理 —— 如果解读放前端调 AI，用户可以改牌阵再求解读，整套就废了。
牌一旦抽出，解读必须基于**服务端记住的那副牌**。

---

## 7. Prompt 设计要点

统一放 `prompts.py`，便于全站调性一致。

塔罗解读的 prompt 结构：

- **System**：设定「占卜师」口吻 —— 象征性反思，不说教、不像客服念稿。
- **护栏**（产品调性，非法律要求）：
  - 不预测生死、不开医疗 / 投资 / 法律的具体建议，遇到就转向「象征层面的提醒」。
  - 不承诺「一定会发生」，用「这张牌在提示……」的说法。
- **输入**：只给三张牌的 `keywords_zh` + 简短牌义 + 用户问题。
  **不喂整副牌** —— 省 token，也避免模型跑偏。
- **输出**：要求结构化 JSON
  ```json
  { "past": "...", "present": "...", "future": "...", "summary": "..." }
  ```
  用 JSON mode / 明确 schema，后端好解析，前端好排版。
- 全中文输出（用户中文提问时）。

**待定，想听你意见**：解读风格偏「温柔共情」还是「直白犀利」？这决定 prompt 怎么写。

---

## 8. 成本与护栏

1. **估算单价表**放 `usage.py`，按模型写死每 M token 价格，算出 `cost_usd`。
   DeepSeek 很便宜，塔罗单次约 1–1.5k output token，成本可忽略，但**可量化**才有意义。
2. **全局日上限**：比如全站一天 500 次，超了返回「今日解读已约满」。
   防有人刷着玩烧额度。这个上限也走 `ai_usage` 计数。
3. **key 管理**：`DEEPSEEK_API_KEY` 走 EC2 环境变量，**不进 Git**。
   `.gitignore` 已有 `Key/` 段，建议再加 `.env`。绝不再犯 `SECRET_KEY` 硬编码的错。
4. **每 feature 独立预算**：将来某个功能烧得凶，能在 `quota.py` 单独设上限，不影响别人。

---

## 9. 从「模块」升级到「独立服务」的路径

现在抽干净接口，是为了将来能无痛拆：

- **触发条件**：出现第二个后端 / 多语言调用方；或 AI 调用重到要独立扩缩容；或别的项目也要共用。
- **拆的时候**：`Backend/Service/ai/` 整个搬出去，`service.py` 的 `run()` 变成 HTTP 端点，
  调用方把 `ai.run(...)` 换成一次 HTTP 调用。**业务 prompt 和配额逻辑原样保留。**
- 所以现在**不要**写任何「假设自己在进程内」的东西：不共享数据库连接对象给功能代码，
  不依赖全局可变状态。功能代码只经 `ai.run` 出入。

---

## 10. 分期建议

| 阶段 | 内容 | 说明 |
|------|------|------|
| P0（本期） | `ai/` 模块骨架 + `ai_service.py` 统一入口 + `ai_usage` 表 + 塔罗 `interpret_tarot` | 半天到一天，能跑通塔罗 |
| P1 | `tarot_readings` 业务表 + `POST /api/tarot/reading` 路由接前端 | 塔罗功能完整闭环 |
| P2 | 全局日上限 + 按 role 差异化配额 + 成本看板（管理页读 `ai_usage`） | 运营需要时 |
| P3 | 第二个 AI 功能接入，验证抽象是否够用 | 真正的验收 |

---

## 11. 需要 Horizon 拍板的点

1. **要不要按此方案做？** 还是先只给塔罗单独写一版、不上层。
2. **解读口吻**：温柔共情 / 直白犀利？
3. **配额**：`user` 一天 1 次；`vip`/`svip` 要不要更多？`horizon` 无限？
4. **全局日上限**设多少？
5. **是否现在就建 `tarot_readings` 表**（P0 就建 vs P1 再建）。

拍完这几个，我可以按 P0 直接开写草稿，先不接现有代码，你审完再合。
