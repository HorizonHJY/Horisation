# AI Service Layer — 设计方案

状态：**已评审，待实现（P0）**
作者：小思（草案 2026-09-20）
评审：Claude（2026-09-20），修订处标 **【评审修订】**；新增 §12 历史 / 打分 / 训练数据

> 目标是让 Horizon 决定「要不要做、做成什么样」，再动手。所有代码路径均为建议，尚未写入仓库。
> 评审后 Horizon 已拍板：**按此方案做**。§11 记录了每个待定项的答案。

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
- 多 provider **路由**（按功能/成本自动选厂商）。接口做成厂商无关（§2），实现先只接一个。
- 向量检索 / RAG（牌义目前是静态小数据，直接塞进 prompt 即可）。
- 流式输出（SSE）。eventlet 单 worker 下要单独验证，v1 用同步 + 明确的等待文案。

---

## 2. 总体形态

**不做独立 App，先在 Horisation 后端内部抽一个模块。**

```
Backend/
├── Service/
│   ├── ai/
│   │   ├── __init__.py        # 暴露单例 ai
│   │   ├── client.py          # 厂商无关的 chat() —— 唯一碰网络的地方
│   │   ├── service.py         # AIService：quota → 调用 → 重试 → 记账 → 降级
│   │   ├── quota.py           # 配额策略（按 feature/role/day）+ 全局日上限
│   │   ├── usage.py           # 用量与成本落库；单价表
│   │   ├── prompts/           # 【评审修订】一个功能一个文件，不是一个 prompts.py
│   │   │   ├── __init__.py
│   │   │   └── tarot.py       # TAROT_PROMPT_VERSION + system/user 模板
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
    quota_key=f"user:{current_username}:tarot:{ai.today_key()}",
    payload={'question': q, 'spread': spread},
)
# result.ok / result.text / result.data / result.usage / result.quota / result.error
```

**关键点：功能代码不直接 import `client.py`。** 谁都不许绕过 service 直接调网络，
否则限流和记账就漏了。

**【评审修订】`client.py` 不以厂商命名，接口厂商无关：**

```python
def chat(messages, *, model, max_tokens, temperature, json_mode=False) -> ChatResponse
# ChatResponse(text, prompt_tokens, completion_tokens, model, latency_ms)
```

厂商由环境变量 `AI_PROVIDER`（`deepseek` / `anthropic`）选择。两家差的只是 URL、
header 和响应字段名——换厂商改 `client.py` 一个文件，`service.py` 以上全部不动。
**不引厂商 SDK，直接用 `requests` 调 HTTP**，原因见 §8.5。

---

## 3. 核心接口契约

### 3.1 统一返回结构

所有 AI 功能，无论内部用哪个模型，返回同一个形状：

```python
@dataclass
class AIResult:
    ok: bool
    text: str | None            # 成功时的正文（markdown 允许）
    data: dict | None           # 结构化输出时（如塔罗分成 past/present/future）；解析失败为 None
    error: str | None           # 失败原因（给用户看的短句，不泄内部细节）
    usage: AIUsage              # tokens / model / cost / latency_ms / attempts
    quota: QuotaState           # used / limit / remaining / reset_at
```

前端只认这一个结构，不再解析各家 API 的奇怪返回。

### 3.2 配额标识

限流规则**不写死在 AI 层**。调用方传一个 `quota_key`，AI 层只负责计数：

```
user:zwy:tarot:2026-09-20
```

- 塔罗：一天 N 次 → key 含日期，天然「跨天自动重置」。
- 将来不限次的功能：传 `None`，跳过额度检查，仅记账。
- 高等级用户配额不同 → 由 `quota.py` 按 role 查上限表（§11 第 3 问）。
- **【评审修订】日期只能由 `ai.today_key()` 产生**（§5）。调用方不许自己 `date.today()`。

### 3.3 调用者身份

每次调用必须带 `feature` 名（`'tarot'` / 以后的 `'listing-helper'` 等）。
用途：日志分类、成本按功能拆分、按功能设独立预算。**没有 feature 名直接报错**，
防止有人图省事不标。

---

## 4. 数据模型

新增两张表（`_data/market.db`）。**只增不改**，不动现有表。

**【评审修订】表的归属现在就划清**，将来拆服务时知道哪张跟着走：

| 表 | 归属 | 拆服务时 |
|---|---|---|
| `ai_usage` | AI 层（基础设施记账） | 跟 `Backend/Service/ai/` 一起搬走 |
| `tarot_readings` | 业务（塔罗功能的数据） | 留在应用里 |

### 4.1 `ai_usage` — 记账与限流的唯一事实来源

```sql
CREATE TABLE ai_usage (
  id           TEXT PRIMARY KEY,          -- uuid4
  feature      TEXT NOT NULL,             -- 'tarot'
  username     TEXT NOT NULL,
  quota_key    TEXT,                      -- 'user:zwy:tarot:2026-09-20'，可为 NULL
  model        TEXT NOT NULL,             -- 'deepseek-chat'
  prompt_version TEXT,                    -- 【评审修订】'tarot-v1'，成本/质量按版本可比
  prompt_tokens      INTEGER DEFAULT 0,
  completion_tokens  INTEGER DEFAULT 0,
  cost_usd     REAL DEFAULT 0,            -- 按当次单价算出的估算成本，写入后不再重算
  latency_ms   INTEGER,
  attempts     INTEGER DEFAULT 1,         -- 【评审修订】含自动重试
  ok           INTEGER NOT NULL,          -- 1/0
  error_kind   TEXT,                      -- 'timeout' / 'rate_limit' / 'provider' / 'parse' / NULL
  created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_ai_usage_quota ON ai_usage(quota_key, ok);
CREATE INDEX idx_ai_usage_feature_day ON ai_usage(feature, created_at);
```

**一个必须先想清楚的设计取舍：失败要不要记账？**

**成功和失败都写 `ai_usage`，但限流只数 `ok=1` 的行。**

- 失败（超时 / provider 报错）**不消耗**用户当天额度 → 用户能重试。这是对的，
  不能让一次网络抖动把用户名额吃掉。
- 但失败**要留下记录** → 否则你永远不知道 AI 到底失败率多高、钱有没有白烧。

**【评审修订】"成功"的定义：模型返回了内容、产生了 usage（token 计了费）就算 `ok=1`**，
即使随后 JSON 解析失败（`error_kind='parse'`，但仍扣额度——钱已经花了，且用户拿到了原文）。
这样"额度"和"成本"的定义一致。配额被拒（还没调用）**不写** `ai_usage`——没有成本，
写进去会污染统计。

限流查询：
```sql
SELECT COUNT(*) FROM ai_usage WHERE quota_key = ? AND ok = 1;
```

### 4.2 `tarot_readings` — 业务内容（**P0 必建**，见 §6）

```sql
CREATE TABLE tarot_readings (
  id             TEXT PRIMARY KEY,          -- 抽牌时生成，前端只拿这个 id 来要解读
  username       TEXT NOT NULL,
  spread_json    TEXT NOT NULL,             -- 抽到的三张牌 + 位置，/draw 时写入
  question       TEXT,                      -- 用户要解读时才填，可为 NULL
  reading_raw    TEXT,                      -- 模型原文，一字不改（训练/评估用）
  reading_json   TEXT,                      -- 解析后的结构；解析失败为 NULL
  model          TEXT,                      -- 冗余自 ai_usage，方便按模型筛
  prompt_version TEXT,                      -- 冗余自 ai_usage，方便按版本筛
  rating         INTEGER,                   -- 1–5，贴合度；NULL = 未评
  rating_note    TEXT,                      -- 可选一句话
  rated_at       TIMESTAMP,
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 抽牌时间
  read_at        TIMESTAMP                  -- 解读时间；NULL = 只抽了牌没要解读
);
CREATE INDEX idx_tarot_user_day ON tarot_readings(username, created_at);
```

为什么要单独一张：`ai_usage` 是「基础设施记账」，`tarot_readings` 是「业务数据」。
塔罗以后要「回看历史牌阵」、做分享页、做训练集，靠这张表。分开存，职责干净。

**【评审修订】`/draw` 就写这张表。** 每次抽牌一行（`read_at` 为 NULL），要解读时更新同一行。
抽了不解读的行会有很多——没关系，它们本身也是数据（多少人抽了却没问）。

---

## 5. 时区 —— 必须现在定死

**这是最容易上线后才炸的坑。**

`date('now')` 是 UTC。用户在美国，本地晚上 8 点抽牌，UTC 已是次日 —— 额度会在
本地时间下午/傍晚被错误重置。两种做法：

- **方案 i（采用）：`quota_key` 里直接写业务时区日期。**
  Python 端用 `zoneinfo` 算 America/Chicago 的日期，拼进 key。
  数据库不参与日期运算，逻辑全在一处，改时区只改一个常量。
- 方案 ii：SQL 里用 `date(created_at, '-5 hours')`。硬编码偏移，遇到夏令时会错，**不采用**。

**【评审修订】时区知识只存在一处：**

```python
# ai/service.py
BUSINESS_TZ = ZoneInfo("America/Chicago")
def today_key() -> str:
    return datetime.now(BUSINESS_TZ).date().isoformat()
```

所有 `quota_key` 里的日期**只能**来自 `ai.today_key()`。调用方自己算日期，早晚有一个忘了时区。

---

## 6. 调用流程（塔罗）

**【评审修订】原稿在解读请求里重新抽牌，与线上流程冲突。** 线上塔罗是：点 Start 时
`/draw` 由服务端抽好三张 → 用户从 78 张里自己点选 → 逐张翻开 → 释义。解读是**翻开之后**
的可选第二步。所以解读接口**不抽牌**，它指向服务端已经抽过、已经落库的那一副。

```
POST /api/tarot/draw                                （现有，改动：落库）
  1. 登录校验
  2. 服务端 secrets 抽三张（现有逻辑）
  3. INSERT tarot_readings(id, username, spread_json)       ← 新增
  4. 返回 { spread, reading_id }                             ← 新增 reading_id
  抽牌不计额度、不花钱，随便抽。

POST /api/tarot/reading                              （新增）
  body: { reading_id, question? }
  1. 登录校验
  2. 取 tarot_readings[reading_id]；不存在或不属于当前用户 → 404
  3. 已有 reading_json → 直接返回，不调模型、不扣额度            ← 幂等，防连点
  4. 校验 question：可选；有则 ≤ 200 字、去控制字符
  5. quota_key = f"user:{u}:tarot:{ai.today_key()}"
  6. ai.run(feature='tarot', quota_key=…, payload={question, spread})
       ├ 超额 → 429 + 友好提示，不调用
       ├ 全局日上限 → 503 + 「今日解读已约满」
       ├ 超时/5xx → 自动重试一次；仍失败 → ai_usage(ok=0)，返回可重试错误，不扣额度
       └ 成功 → ai_usage(ok=1)；解析 JSON，失败则 data=None 仍返回原文
  7. UPDATE tarot_readings SET question, reading_raw, reading_json, model,
       prompt_version, read_at
  8. 返回 { reading: {text, data}, quota }

POST /api/tarot/readings/<id>/rating                 （新增，§12）
GET  /api/tarot/readings                             （新增，§12，本人历史）
```

**为什么抽牌和解读必须绑在服务端记住的那副牌上：**
现有 `tarot_controller.py` 注释已经写明「能在 devtools 重抽的牌阵不算牌阵」。
同理 —— 如果解读接口接受前端传来的牌，用户可以改牌阵再求解读，整套就废了。
`reading_id` 就是那条线。

---

## 7. Prompt 设计要点

`prompts/tarot.py`，导出 `TAROT_PROMPT_VERSION = "tarot-v1"`（每次实质改动递增，
写进 `ai_usage` 和 `tarot_readings`，成本、失败率、打分才能按版本比较）。

**口吻（§11 第 2 问已定）：温和但具体。** 不预言、不说教、不像客服念稿；把每张牌落到
一件真实的事上；结尾给**一个**明天就能做的小动作。象征层面的提醒，不是判词。

- **System**：占卜师角色 + 口吻边界 + 硬规则：
  - 不预测生死、不给医疗 / 投资 / 法律的具体建议，遇到就转向象征层面。
  - 不承诺「一定会发生」，用「这张牌在提示……」的说法。
  - 不逐张报「这张牌代表 X」——位置隐含，不宣告。
- **输入**：只给三张牌的 `name` / `name_zh` / `keywords_zh` + 简短牌义 + 用户问题（可选）。
  **不喂整副牌** —— 省 token，也避免模型跑偏。牌义标明是「原料，不要照抄措辞」。
- **输出**：JSON mode，固定 schema：
  ```json
  { "past": "...", "present": "...", "future": "...", "summary": "...", "next_step": "..." }
  ```
  每个位置两三句；`summary` 是整体；`next_step` 是那一件小事。
- **语言**【评审修订】：跟随用户问题的语言；**没有问题时默认中文**。牌名中英并列。
- **没有问题时**：读「此刻的处境」，不要替用户编一个问题。

结构参考了几个公开项目的做法（TarocchAI 的原则、tarotAI 的 schema），**文字是我们自己的**——
其中写得最好的那个仓库没有许可证，不能逐字用。

---

## 8. 成本与护栏

1. **估算单价表**放 `usage.py`，按模型写死每 M token 价格，**旁注核对日期**。
   `cost_usd` 在调用当时算出并写入，之后调价不回改历史。
2. **全局日上限**【评审修订：P0 就做，不等 P2】：全站一天 **100** 次（§11 第 4 问），
   超了返回「今日解读已约满」。11 个人绰绰有余，同时是硬安全网。五行代码。
3. **`AI_ENABLED=0` 总开关**【评审修订】：环境变量，P0 就加。key 泄露或费用异常时一秒关掉，
   比日上限更早需要。关掉时 `ai.run` 直接返回 `ok=False, error='AI 功能暂时关闭'`。
4. **key 管理**：`DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` 走 EC2 环境变量或 `Key/` 目录
   （已 gitignore），**不进 Git**。绝不再犯 `SECRET_KEY` 硬编码的错。
5. **【评审修订】eventlet 单 worker 的硬约束**：生产是 `gunicorn -k eventlet -w 1`——
   单进程、协作式。一次 AI 调用 5–30 秒，如果用的 HTTP 库不被 eventlet 打绿（monkey-patch），
   这 30 秒**全站冻结**：聊天、socket、所有人。`requests` 会被打绿；厂商 SDK 底层若是 `httpx`
   则未必。所以 **`client.py` 用 `requests` 裸调 HTTP，不引 SDK**，并硬性 `timeout=(5, 30)`。
   本地 `LOCAL_DEV=1` 是 threading 模式，不会暴露这个问题——**只有生产会**。
6. **每 feature 独立预算**：将来某个功能烧得凶，能在 `quota.py` 单独设上限，不影响别人。
7. **用户输入进 prompt**：问题限 200 字，去控制字符。有 system prompt + JSON schema 兜着，
   注入的最坏结果是一段奇怪的解读，可接受。

---

## 9. 从「模块」升级到「独立服务」的路径

现在抽干净接口，是为了将来能无痛拆：

- **触发条件**：出现第二个后端 / 多语言调用方；或 AI 调用重到要独立扩缩容；或别的项目也要共用。
- **拆的时候**：`Backend/Service/ai/` 整个搬出去（含 `ai_usage` 表），`service.py` 的 `run()`
  变成 HTTP 端点，调用方把 `ai.run(...)` 换成一次 HTTP 调用。**业务 prompt 和配额逻辑原样保留。**
- 所以现在**不要**写任何「假设自己在进程内」的东西：不共享数据库连接对象给功能代码，
  不依赖全局可变状态。功能代码只经 `ai.run` 出入。
- **【评审修订】测试**：`AIService` 构造时接受注入的 client，测试用 fake client 不联网。
  时区、配额计数、JSON 解析兜底、幂等——这四样正是"上线后才炸"的那种，P0 就要有测试。
  后端目前零测试，这一层开这个头。

---

## 10. 分期（评审后）

| 阶段 | 内容 | 说明 |
|------|------|------|
| **P0（本期）** | `ai/` 模块 + `ai_usage` + `tarot_readings` + `/draw` 落库 + `/reading` + 打分接口 + 前端"整体解读"一节（含可选问题框、打分） + 总开关 + 全局日上限 + 测试 | 约一天，直接闭环。原 P1 并入 |
| P1 | `GET /api/tarot/readings` 本人历史 + 前端历史页 | 表已经在了，只是加读接口和页面 |
| P2 | 按 role 差异化配额 + 成本看板（管理页读 `ai_usage`） | 运营需要时 |
| P3 | 第二个 AI 功能接入，验证抽象是否够用 | 真正的验收 |
| P4 | 训练/评估数据导出脚本（§12.3） | 攒够数据再说 |

---

## 11. 已拍板

| 问题 | 答案 |
|---|---|
| 1 按此方案做？ | **做**，含 §6 修正 |
| 2 解读口吻 | **温和但具体**：不预言、不说教，落到「明天可以做的一件小事」 |
| 3 配额 | `user` 1/天，`vip` `svip` 3/天，`admin` `horizon` 不限但照常记账。全在 `quota.py` 一张字典 |
| 4 全局日上限 | **100**，P0 就加 |
| 5 `tarot_readings` | **P0**，`/draw` 就写它 |

---

## 12. 历史、打分、训练数据（Horizon 追加，2026-09-20）

### 12.1 存什么

每次解读完整留档：**问题 + 牌阵 + 模型原文 + 解析后结构 + 模型/prompt 版本 + 用户打分**。
全部在 `tarot_readings` 一张表里（§4.2 已按此设计），不另开表。

`reading_raw`（原文）和 `reading_json`（解析后）分开存，理由：训练/评估要的是模型说了什么，
不是我们解析成了什么；而且解析规则将来会改，原文不会。

### 12.2 打分

解读展示出来之后，下面一行：**「贴合吗？」+ 1–5 星 + 可选一句话**。

```
POST /api/tarot/readings/<id>/rating   body: { rating: 1-5, note?: ≤ 100 字 }
```

- 只能给自己的解读打分；可以改分（覆盖，`rated_at` 更新）。
- 不打分不影响任何功能。前端不弹窗催，就放在那儿。
- 打分是**产品信号**，不是用户画像：只用来评估 prompt 版本和模型，不用来给用户分类。

### 12.3 将来做训练/评估

现在什么都不做，只保证**数据形状够用**：

- 按 `prompt_version` 分组看平均分 → 知道改 prompt 有没有变好。这是最先有价值的用法，
  几十条数据就能看出趋势，不需要"训练"。
- 攒够量之后，导出脚本把 `(question, spread_json, reading_raw, rating)` 写成 JSONL，
  高分样本做 few-shot 示例或微调集，低分样本做失败分析。脚本放 `scripts/`，P4。
- **隐私边界**：问题是用户写给塔罗的私事。数据留在本站数据库里（和聊天记录同级）；
  导出做训练是站长的决定，导出前**去掉 `username`**。这条写在这里，将来做的人看得见。

### 12.4 历史页（P1）

`GET /api/tarot/readings`：本人的，按时间倒序，分页。前端一页：日期、三张牌缩略、
问题一行、点开看解读和当时的打分。**只看自己的**——别人的问题不给看，包括 `horizon`
在界面上也不看（要看走数据库/MCP，那是运维不是产品）。

---

## 13. 评审摘要（Claude，2026-09-20）

原稿的分层、契约、`quota_key` 外置、失败记账不扣额、`zoneinfo`、"只经 `ai.run` 出入"——
全部保留。修订的是：

1. **§6 解读接口不抽牌**，改为引用 `/draw` 落库的 `reading_id`——和线上流程一致，堵死改牌阵求解读，白送幂等。由此 `tarot_readings` 必须 P0。
2. **§8.5 eventlet 单 worker**：`client.py` 用 `requests`，不引 SDK，硬超时。这个库特有，原稿没提。
3. `client.py` 厂商无关；`prompts/` 包；表的归属划线；`today_key()` 收口时区；JSON 解析兜底；自动重试一次；`AI_ENABLED` 总开关；全局日上限提前到 P0；测试。
4. §12 历史 / 打分 / 训练数据是 Horizon 在评审后追加的需求，表结构已按此调整。
