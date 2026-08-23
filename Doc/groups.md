# Horisation — Groups（群组）设计文档

Last updated: 2026-08-22

---

## 需求（P0）

独立建组、按用户拉人进组、群聊。**与好友关系无关** —— 不要求互为好友即可拉人（按用户 id/用户名搜索添加）。

一句话：用户可创建群，通过搜索用户名把任意平台用户拉进群，群里可实时群聊。

## 概念边界

- 独立概念：群组成员 ≠ 好友关系。拉人只校验用户存在，不校验 friendship。
- 建组者是 owner（可踢人/改群名/解散群）；成员可退出群。
- 与 `friendships` / `friend_requests` 完全解耦。

---

## 数据结构（DB）

新增两张表（追加到 `market_db.py` 的模型区 + `Doc/data_storage.md`）。

### Table: `groups`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| name | TEXT | 群名，max 50 字符 |
| owner | TEXT | 建组者 username |
| created_at | DATETIME | UTC |

### Table: `group_members`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| group_id | TEXT (UUID) | FK → groups.id (cascade delete) |
| username | TEXT | 成员 username，indexed |
| role | TEXT | `owner` / `member` |
| joined_at | DATETIME | UTC |

> `group_id + username` 唯一（一名成员在一组只能有一条记录）。owner 同时是 member 表里 role=owner 的一行。

### 群聊消息表 `group_messages`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| group_id | TEXT (UUID) | FK → groups.id (cascade delete)，indexed |
| sender | TEXT | 发送者 username |
| content | TEXT | max 1000 chars |
| created_at | DATETIME | UTC |

---

## API 设计（blueprint: `/api/groups`）

| Method | Path | 说明 | 权限 |
|--------|------|------|------|
| GET | `/api/groups` | 我加入的群列表 | login_required |
| POST | `/api/groups` | 建组 `{name}`，建完自动加入(owner) | login_required |
| GET | `/api/groups/<id>` | 群详情 + 成员列表 | 成员 |
| PUT | `/api/groups/<id>` | 改群名 `{name}` | owner |
| DELETE | `/api/groups/<id>` | 解散群 | owner |
| GET | `/api/groups/<id>/messages` | 群聊历史 | 成员 |
| POST | `/api/groups/<id>/messages` | 发消息 `{content}` | 成员 |
| POST | `/api/groups/<id>/members` | 拉人 `{username}` | owner |
| DELETE | `/api/groups/<id>/members/<username>` | 踢人 / 退出 | owner 踢人；本人可退出 |

入参校验：群名非空且 ≤50；消息非空且 ≤1000；username 必须存在且非本人（拉人）。

---

## 前端设计

- 新页面 `frontend/src/pages/Groups.jsx`，路由 `/groups`
- 在 `Sidebar.jsx` 加大纲入口
- 功能：群列表、建群弹窗、进群、拉人搜索（复用 `search_users` 思路）、群聊界面
- 移动端适配：遵循 `Doc/mobile_ux_principles.md`（消息输入区底部固定、列表可滚动）
- 实时性：先做轮询拉取消息（简单可靠），Socket 实时可作为后续增强

---

## 注意事项

- 所有校验放 controller 层，返回 `ok:false` + `error`
- 越权操作（非成员读群、非 owner 踢人）返回 403
- 删除群时级联删除 `group_members` + `group_messages`
- 拉人前校验用户存在且 `is_active`
