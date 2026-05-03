# Horisation — Development Log

## 0. Current Status
Last Updated: 2026-05-02

### Current Working Version
- **Completed**: 邀请码系统 + 公开注册、功能角色门控（Online Gomoku）、市集商品编辑、Profile 地址字段扩展、好友/私信/联系方式申请系统、用户数据 SQLite 迁移、二手市集、留言板、移动端响应式侧边栏、React SPA 全面迁移
- **In Progress**: [待补充]
- **Blocked / Not Solved**: 密码明文存储（待迁 bcrypt）；无 CI/CD 流水线

### Latest Summary
新增了邀请码机制实现受控公开注册，Online Gomoku 按角色门控，市集支持编辑商品，Profile 扩展了地址和邮编字段。

### Next Immediate Step
[待补充 — 请在下次迭代前更新]

---

## 1. Technical Decisions

| Date | Decision | Reason | Trade-off |
|---|---|---|---|
| 2026-03-01 | Flask 纯 API-only + React SPA，彻底放弃 Jinja2 | 前后端解耦，便于独立迭代；React 生态更利于构建交互界面 | 首次部署需要额外构建步骤；本地开发需要同时跑 Flask + Vite 两个进程 |
| 2026-03-01 | 通过 `ProxyFix` 解决 HTTPS cookie 问题 | Nginx 反代后 Flask 看到的是 HTTP 请求，无法设置 Secure Cookie | 需要信任 `x_for=1` 层，多层代理时要调整参数 |
| 2026-03-02 | 图片存 Cloudflare R2，不存服务器本地 | EC2 磁盘有限，R2 成本低且全球分发；与 S3 API 兼容，迁移方便 | 需要额外管理 R2 凭证（gitignore）；上传失败需处理 R2 回滚 |
| 2026-03-02 | 市集数据用 SQLite + SQLAlchemy，独立 `market.db` | 比 JSON 文件更可靠；SQLAlchemy ORM 便于将来迁移到 PostgreSQL | SQLite 不支持高并发写；`market.db` gitignore，需手动维护生产数据库 |
| 2026-03-06 | 用户/Session 从 JSON 文件迁移到 SQLite | JSON 文件非线程安全，并发写有数据损坏风险；统一数据层 | 迁移脚本需要一次性执行，生产环境需人工确认迁移成功 |
| 2026-03-06 | SocketIO 本地用 threading，生产用 eventlet + Redis | 本地开发不需要 Redis 依赖，降低启动复杂度；生产需要跨进程广播 | 两套配置通过 `LOCAL_DEV=1` 环境变量区分，需保持同步 |
| 2026-03-06 | 好友关系与联系方式申请分开设计（两张独立表） | 好友 = 可以聊天；看联系方式 = 额外授权，保护用户隐私 | 逻辑略复杂，前端需处理两类请求状态 |
| 2026-05-02 | 邀请码限时生效（valid_from / valid_to），非单次消耗型 | 便于为活动窗口批量分发邀请，无需逐人管理 | 同一邀请码可被多人使用，无法精确控制注册人数上限 |

---

## 2. Reusable Patterns / Lessons Learned

### Pattern 1: Flask 在反向代理后 Secure Cookie 失效
- **Symptom**: 生产环境登录后 session cookie 不生效，服务器日志显示 Flask 将请求识别为 HTTP
- **Root Cause**: Nginx 反代后，Flask 收到的请求协议为 HTTP（内部通信），`SESSION_COOKIE_SECURE=True` 拒绝在非 HTTPS 连接上设置 cookie
- **Reusable Solution**: 在 `app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)` 让 Flask 信任代理头；同时用 `LOCAL_DEV=1` 环境变量在本地关闭 `SESSION_COOKIE_SECURE`

### Pattern 2: Python 3.9 不支持 `X | Y` 联合类型注解
- **Symptom**: 服务器部署后出现 502，本地 Python 3.11 完全正常
- **Root Cause**: `dict | None` 语法是 Python 3.10+ 的特性；EC2 当时运行 Python 3.9
- **Reusable Solution**: 跨版本兼容时统一用 `Optional[dict]`（需 `from typing import Optional`）；或直接升级 Python 版本（最终方案）

### Pattern 3: JSON 文件存储用户数据的隐患
- **Symptom**: 并发请求偶发数据写入丢失；`users.json` 提交 git 引发冲突
- **Root Cause**: 文件级读写没有锁，多进程/多线程并发时会产生竞态；敏感数据不应进 git
- **Reusable Solution**: 用 SQLite（单机）或 PostgreSQL（多实例）替代 JSON 存储；`.gitignore` 所有含用户数据的文件

### Pattern 4: 用户查找必须用字段匹配，不能用 dict key
- **Symptom**: 特定用户名无法登录，日志显示 `authenticate_user` 返回失败
- **Root Cause**: `users.json` 的 dict key（如 `"user_001"`）与 `username` 字段值不一致；代码直接用 key 查找
- **Reusable Solution**: 永远用 `_find_user()` helper 按 `username` 字段搜索，不依赖存储结构的 key

---

## 3. Iteration History

---

### 2026-05-02 — 邀请码注册、功能门控、市集编辑、Profile 扩展

#### Goal
实现受控的公开注册（邀请码机制）；把 Online Gomoku 限定为特定角色可用；允许用户编辑已发布的市集商品；扩展 Profile 联系信息字段（地址、邮编）。

#### Trigger / Context
平台逐渐成熟，需要允许朋友自助注册而不是全部由管理员手动创建；Gomoku 功能仍在灰度，需要角色门控；市集商品发布后无法修改是明显的体验缺陷。

#### Problem & Root Cause
无明显 bug，本次为功能开发与结构优化。

各子功能动机：
- 邀请码：admin 手动创建用户效率低，但完全开放注册不符合"私密平台"定位
- Gomoku 门控：之前所有登录用户均可访问，但功能尚未完善
- 市集编辑：`PUT /api/market/listings/<id>` 端点已有但缺少 `original_price` 支持，前端也没有 EditModal
- Profile 地址：用户在市集交易时需要填写收货地址，Profile 之前没有此字段

#### Solution

**邀请码系统**
- `market_db.py` 新增 `InviteCode` 模型（`invite_codes` 表），字段：`code`, `valid_from`, `valid_to`, `created_by`，启动时自动建表
- 新增后端端点（仅限 `horizon` 用户）：`GET/POST /api/auth/invite-codes`、`DELETE /api/auth/invite-codes/<id>`
- 新增公开端点 `POST /api/auth/signup`：校验邀请码时效性，注册成功后自动登录，分配 `user` 角色
- `Register.jsx`：与 Login 页同款 FlowerCanvas 背景，字段含 username/display name/password/confirm/invite code
- `Login.jsx` 底部加"Register with invite code"链接
- `App.jsx` 注册 `/register` 为 `PublicOnlyRoute`，新增 `FeatureRoute` 守卫组件

**功能门控**
- `features.js`：新增 `onlineGomoku` flag，白名单 `['horizon', 'horizonadmin', 'vip3']`
- `Sidebar.jsx`：`For Fun` 区块在用户无可见条目时整体隐藏；Online Gomoku 条目按 `useFeature` 过滤
- `App.jsx`：路由用 `<FeatureRoute>` 包裹，未授权角色直接跳转 `/home`

**市集编辑**
- `market_db.py`：`update_listing()` 增加 `original_price` 字段支持
- `market_controller.py`：PUT 端点解析并校验 `original_price`（接受空字符串以清空）
- `Market.jsx`：新增 `EditModal` 组件，支持编辑标题/描述/原价/售价/分类；`handleEditSave` 调用 PUT 后刷新列表

**Profile 扩展**
- `User` 模型新增 `address`（Text）和 `postal_code`（String 20）列
- `_migrate_columns()` 自动在现有 DB 上添加新列（幂等）
- `user_manager.update_user_profile()` 和 `PUT /api/auth/profile` 均接受并持久化新字段
- `Profile.jsx`：联系信息区块新增 Address（col-8）和 Postal Code（col-4）输入框

**Profile UI 小修**
- 移除"Permission level: X"显示（内部实现细节，不应对用户可见）
- Avatar 文件选择器改为隐藏 `<input>` + 自定义"Upload Photo"按钮，消除浏览器本地化"选择文件"标签

#### Changed Files
- `Backend/Controller/market_db.py` — 新增 `InviteCode` 模型及 helpers；`_migrate_columns()` 新增地址列；`update_listing()` 支持 `original_price`
- `Backend/Controller/auth_controller.py` — 新增 `/signup`、`/invite-codes` CRUD 端点
- `Backend/Controller/market_controller.py` — PUT 端点支持 `original_price`
- `Backend/Controller/user_manager.py` — `update_user_profile()` 支持 `address`、`postal_code`
- `frontend/src/features.js` — 新增 `onlineGomoku` feature flag
- `frontend/src/App.jsx` — 新增 `FeatureRoute` 组件；注册 `/register` 路由
- `frontend/src/pages/Login.jsx` — 添加注册入口链接
- `frontend/src/pages/Register.jsx` — 新建文件，公开注册页面
- `frontend/src/pages/Market.jsx` — 新增 `EditModal`，Edit 按钮，`handleEditSave`
- `frontend/src/pages/Profile.jsx` — 新增地址/邮编字段；avatar 上传 UI 重构；移除权限等级显示
- `frontend/src/components/Sidebar.jsx` — For Fun 区块条件渲染；Online Gomoku 按 feature flag 过滤

#### Result
- 用户可通过有效邀请码自助注册，无需管理员介入；`horizon` 用户可在 Admin 页管理邀请码的有效期
- Online Gomoku 对未授权角色不可见且 URL 直接访问被重定向
- 市集商品可编辑（含原价字段）
- Profile 可保存地址和邮编信息

#### Testing
- 使用有效邀请码在 `/register` 完成注册，验证自动登录、分配 `user` 角色；使用过期邀请码验证 403 拒绝
- 用 `user` 角色账号访问 `/fun/gomoku-online`，验证重定向到 `/home`；用 `horizon` 账号验证正常访问
- 在市集发布一条商品，通过 Edit 按钮修改标题和原价，验证更新后列表数据正确
- 在 Profile 填写地址和邮编，保存后刷新页面验证持久化

#### Lessons Learned
- **Symptom**: 希望部分功能只对特定角色开放
- **Root Cause**: 路由和侧边栏没有权限感知机制
- **Reusable Solution**: 用 `features.js` 维护角色白名单，`useFeature(flag)` hook 同时保护 UI 渲染和路由访问；后端对应端点仍需独立校验，不能只依赖前端门控

#### Remaining Issues / Next Step
- 密码仍为明文存储，需迁移 bcrypt
- 邀请码当前为"时间窗口内无限次使用"，若需限制注册人数需加 `max_uses` 字段
- [待补充下一步优先事项]

---

### 2026-03-06 — 用户数据迁移至 SQLite + 好友系统 + 市集升级 + 登录页重设计

#### Goal
将用户账号和 Session 从 JSON 文件迁移到 SQLite；实现完整的好友系统（好友申请、私信、联系方式申请）；升级市集体验（卖家信息、Reach Out 按钮）；重设计登录页；新增本地开发脚本；实现在线五子棋。

#### Trigger / Context
`users.json` 非线程安全，随用户量增加存在并发写入风险；多个功能模块都需要更可靠的用户数据层。好友系统是平台"私密社区"定位的核心功能。登录页视觉体验需要提升。

#### Problem & Root Cause
无明显 bug，本次为大规模功能开发与架构升级。

核心驱动：JSON 文件存储方式已不满足平台需求；好友、私信、联系方式申请是社区功能的基础模块；市集缺少社交互动入口。

#### Solution

**用户/Session 迁移**
- `users.json` 和 `sessions.json` 完全退役；由 `market.db` 中的 `user` 和 `session` 表接管
- `User` 模型：自增整数 PK，包含所有原有字段 + `contact_hidden` 布尔值
- `UserSession` 模型：token PK，含 `expires_at` 过期时间
- `_migrate_from_json()`：启动时检测 `users.json` 是否存在，完成迁移后重命名为 `.migrated`（幂等）
- `user_manager.py` 完全重写，公开 API 不变，底层改用 `market_db.py` 的 SQLAlchemy helpers
- `db_search_users(q)`：ilike 模糊搜索 username + display_name，limit 20

**好友系统**
- 新模型：`FriendRequest`, `Friendship`, `PrivateChatMessage`（均在 `market_db.py`）
- 新 Blueprint `friends_controller.py`（`/api/friends/*`），含用户搜索、好友申请 CRUD、好友列表、私信历史等端点
- `friends_socket.py`：Socket.IO 实时通知（好友申请、接受、联系方式申请）
- **未读消息**：`ChatRead` 模型追踪每个用户每个聊天室的最后读取时间；`GET /api/friends/unread` 返回未读计数；`UnreadContext` 每 30 秒轮询；Sidebar 显示红色 badge
- `Friends.jsx`：三 Tab（Friends / Requests / Add）；实时私信（Socket.IO，room key 为排序后的用户名对）；从市集页带参数跳转时自动打开对应聊天

**联系方式申请系统**
- 新模型 `ContactRequest`：独立于好友关系，记录联系方式查看授权
- `contact_hidden` 字段：用户可设置"隐藏联系方式"，隐藏后他人无法发申请
- `GET /<username>/contact` 端点在返回信息前校验 `has_contact_access()`
- `Friends.jsx` 每位好友显示联系方式状态（申请/待审/已获授权/对方已隐藏）
- 实时通知：`notify_contact_request()` 在 DB 写入后通过 Socket.IO 推送，目标用户不需要刷新页面

**市集升级**
- Browse Tab 过滤掉自己的商品
- Tab 顺序调整：Browse → My Listings → Post Item
- 商品卡片显示卖家头像 + 昵称
- 卖家信息 Modal：头像、昵称、全部在售商品、Reach Out / Add Friend 按钮
- "Reach Out" 按钮：已是好友 → 跳转 `/friends` 并预填聊天消息；非好友 → 自动发送带商品信息的好友申请
- 新 API `GET /api/market/user/<username>`
- 列表响应附带 `seller_display`、`seller_avatar`（批量 join 查询，无 N+1）
- `Listing` 模型新增 `original_price` 字段

**登录页重设计**
- 新组件 `FlowerCanvas.jsx`：水彩花瓣生长动画，canvas 实现，ResizeObserver 响应窗口变化
- 登录页：全屏 canvas 背景 + 毛玻璃登录卡片 + Logo + "Arch Bay" 标题 + 左下角中文标语
- 响应式：`<= 600px` 时 Logo 缩小、标语隐藏

**在线五子棋**
- 新页面 `/fun/online-gomoku`，实时多人对战
- `GameRoom` 模型（`market_db.py`）：存储棋盘 JSON、当前落子方、胜者、获胜格子
- `game_controller.py`：Socket.IO 事件（创建/加入/落子/离开/重置房间）
- `OnlineGomoku.jsx`：大厅列表 + 游戏棋盘 UI

**本地开发**
- `scripts/dev.bat`：一键启动 Flask + Vite 双进程
- `scripts/_flask_local.bat`：注入 `LOCAL_DEV=1` 环境变量
- `app.py`：`LOCAL_DEV=1` 时 SocketIO 切换为 `threading` 模式，关闭 `SESSION_COOKIE_SECURE`

#### Changed Files
- `Backend/Controller/market_db.py` — 新增 `User`, `UserSession`, `FriendRequest`, `Friendship`, `PrivateChatMessage`, `ContactRequest`, `ChatRead`, `GameRoom` 模型及所有 helper 函数
- `Backend/Controller/user_manager.py` — 完全重写，底层改 SQLAlchemy
- `Backend/Controller/friends_controller.py` — 新建，好友系统 API Blueprint
- `Backend/Controller/friends_socket.py` — 新建，好友/联系方式实时通知
- `Backend/Controller/game_controller.py` — 新建，五子棋 Socket.IO 事件
- `Backend/Controller/socketio_instance.py` — 新建，共享 SocketIO 实例
- `Backend/Controller/market_controller.py` — 新增卖家信息端点；`_enrich_listings()` 批量 join
- `app.py` — 注册新 Blueprint；SocketIO 双模式初始化
- `frontend/src/App.jsx` — 新增 `UnreadContext`；注册 `/friends` 路由
- `frontend/src/pages/Friends.jsx` — 新建，好友/私信/申请页面
- `frontend/src/pages/Market.jsx` — 卖家 Modal、Reach Out 按钮、Tab 顺序调整
- `frontend/src/pages/Login.jsx` — 全新设计，使用 FlowerCanvas
- `frontend/src/components/FlowerCanvas.jsx` — 新建，花瓣动画组件
- `frontend/src/components/Sidebar.jsx` — Friends 未读 badge
- `scripts/dev.bat`, `scripts/_flask_local.bat` — 新建
- `scripts/deploy.sh` — 移除 JSON 备份逻辑

#### Result
- 用户数据全面迁移至 SQLite，并发安全性大幅提升；旧 `users.json` 自动归档
- 好友申请、私信、联系方式申请功能完整上线，均支持实时通知
- 市集社交互动闭环：从看商品到联系卖家一键完成
- 登录页视觉大幅升级
- 在线五子棋可在 `horizon`/`horizonadmin`/`vip3` 用户间对战
- 本地开发一键启动，无需手动配置双进程

#### Testing
- 在旧有 `users.json` 存在时启动服务，验证迁移脚本执行、文件重命名为 `.migrated`、用户可正常登录
- 两个账号互相发好友申请、接受、发私信，验证实时通知和未读 badge 更新；刷新页面后验证消息持久化
- 在市集点击"Reach Out"，验证好友/非好友两种路径的跳转和预填消息
- 本地执行 `scripts/dev.bat`，验证 Flask `:5000` 和 Vite `:5173` 均正常启动，Socket.IO 连接成功

#### Lessons Learned
- **Symptom**: 好友关系建立后，目标用户需要刷新页面才能看到新的聊天或联系方式申请
- **Root Cause**: 前端轮询间隔为 30 秒，实时性不足
- **Reusable Solution**: 对需要即时感知的事件（好友申请、消息到达）使用 Socket.IO 推送；轮询只作为兜底容错机制

#### Remaining Issues / Next Step
- 密码仍为明文，需迁 bcrypt
- `contact_hidden` 功能已上线，但 Profile 页面的说明文案可以更清晰
- 邀请码系统（下一个迭代完成）

---

### 2026-03-02 — 二手市集、留言板、用户管理完善、移动端适配

#### Goal
上线二手市集（图片上传到 R2、SQLite 存储商品数据）；上线公共留言板；完善管理员用户管理功能；支持用户自助修改个人信息和头像；优化导航和 UI；适配移动端侧边栏；升级服务器 Python 版本。

#### Trigger / Context
平台基础框架（React SPA + 登录认证）已稳定，开始构建核心功能模块。二手市集是"朋友圈私密平台"的核心场景；留言板提供轻量级社区互动；管理员之前只能创建用户，无法编辑或删除。

#### Problem & Root Cause
两个 bug 在此迭代中修复：

1. **服务器 502**：Python 3.9 不支持 `dict | None` 语法（3.10+ 特性），部署后 Flask 启动失败。根因是本地开发用 Python 3.11，未做版本兼容测试。
2. **Bootstrap Modal 失效**：`index.html` 缺少 Bootstrap JS bundle，Modal 组件无法弹出（Bootstrap 的 JS 交互依赖独立引入）。

#### Solution

**二手市集**
- 新 Blueprint `market_controller.py`（`/api/market/*`）：CRUD 端点 + sold 标记
- `market_db.py`：`Listing` + `ListingImage` SQLAlchemy 模型，数据库路径 `_data/market.db`，启动自动创建
- `r2_manager.py`：封装 boto3 上传/删除，凭证读取自 `Key/r2_config.json`（gitignore）
- 商品最多 3 张图片；删除商品时同步清理 R2 上的图片文件
- `Market.jsx`：Browse / Post Listing / My Listings 三 Tab，图片预览，价格格式化

**留言板**
- `feedback_controller.py`（`/api/feedback/*`）：最新 200 条消息；用户删除自己的消息，管理员删除任意
- `Feedback.jsx`：相对时间显示，500 字符限制，头像/首字母 fallback
- `Message` 模型复用 `market.db`

**用户管理**
- Admin 新增：编辑昵称/邮箱、重置密码、删除用户
- 新端点：`PUT /api/auth/users/<u>/profile`、`PUT /api/auth/users/<u>/password`、`DELETE /api/auth/users/<u>`
- `horizon` 账号受保护，不可删除

**自助 Profile**
- 用户可自助更新昵称、邮箱、密码、头像
- 头像存 R2 `avatars/<username>.<ext>`，覆盖上传
- 新端点：`PUT /api/auth/profile`、`PUT /api/auth/password`、`POST /api/auth/avatar`
- `avatar_url` 加入 session 信息

**导航与 UI**
- Sidebar 新增 Community 区块（Market、Message Board）
- Log Out 固定在底部
- Logo 替换为 `logol.avif`（放置在 `frontend/public/` 由 Vite 伺服）
- Hormemo 移入 Toolkit 区块（对所有角色可见）；CSV Workspace 仅 horizon 可见

**移动端侧边栏**
- `< 768px` 时侧边栏默认隐藏（`translateX(-100%)`）
- Topbar 新增汉堡菜单按钮（仅移动端显示）
- 点击汉堡展开侧边栏，背景遮罩点击关闭；导航后自动关闭
- CSS 媒体查询控制 Topbar 和主内容区全宽展示

**服务器升级**
- Python 3.9 → 3.11，新 venv `/home/ec2-user/venv311/`
- 确认稳定后删除旧 venv

#### Changed Files
- `Backend/Controller/market_db.py` — 新建，`Listing`, `ListingImage`, `Message` 模型
- `Backend/Controller/r2_manager.py` — 新建，R2 上传/删除
- `Backend/Controller/market_controller.py` — 新建，市集 API Blueprint
- `Backend/Controller/feedback_controller.py` — 新建，留言板 Blueprint
- `Backend/Controller/auth_controller.py` — 新增 profile/password/avatar/admin CRUD 端点
- `app.py` — 注册新 Blueprint；`MAX_CONTENT_LENGTH` 100MB
- `frontend/src/pages/Market.jsx` — 新建
- `frontend/src/pages/Feedback.jsx` — 新建
- `frontend/src/pages/Profile.jsx` — 新建
- `frontend/src/pages/AdminUsers.jsx` — 补充编辑/重置密码/删除功能
- `frontend/src/components/Layout.jsx` — 移动端侧边栏状态管理
- `frontend/src/components/Topbar.jsx` — 汉堡按钮
- `frontend/src/components/Sidebar.jsx` — `isOpen`/`onClose` props；Community 区块
- `frontend/src/index.css` — 移动端媒体查询
- `frontend/index.html` — 引入 Bootstrap JS bundle
- `requirements.txt` — 新增 `sqlalchemy`, `boto3`, `flask-socketio`（预置）
- `scripts/deploy.sh` — 新建，标准化部署流程
- `.gitignore` — 新增 `_data/users.json`, `_data/sessions.json`, `_data/market.db`

#### Result
- 二手市集完整上线：发布/浏览/标记已售，图片存 R2
- 留言板上线
- 管理员可完整管理用户生命周期
- 用户可自助更新个人信息和头像
- 移动端可正常使用（侧边栏收纳）
- 服务器稳定运行在 Python 3.11

#### Testing
- 发布包含 3 张图片的市集商品，验证 R2 上传成功、列表展示正确；删除商品验证 R2 图片同步清理
- 用 `user` 角色登录，在留言板发帖/删帖；用 admin 账号删除其他用户帖子
- Admin 页面编辑用户昵称、重置密码后用新密码登录验证；尝试删除 `horizon` 账号验证被拒绝
- 上传头像（2MB 以内 JPEG），Profile 页面刷新后确认头像显示
- 在移动端浏览器（375px 宽）验证汉堡菜单展开/关闭，导航后侧边栏自动收起
- 部署至服务器后验证 502 不再出现（Python 版本升级 + 类型注解修复）

#### Lessons Learned
- **Symptom**: 本地正常，部署后 502
- **Root Cause**: `dict | None` 联合类型注解为 Python 3.10+ 语法，服务器运行 3.9
- **Reusable Solution**: 生产/本地 Python 版本须保持一致；或统一用 `Optional[X]`（`typing` 模块，兼容 3.7+）；CI 中加版本矩阵测试可提前发现

- **Symptom**: Bootstrap Modal 点击无反应
- **Root Cause**: 只引入了 Bootstrap CSS，未引入 Bootstrap JS；Modal/Dropdown 等交互组件依赖 JS
- **Reusable Solution**: 使用 Bootstrap 组件时必须同时引入 `bootstrap.bundle.min.js`（含 Popper）

#### Remaining Issues / Next Step
- 好友和私信系统（下一个迭代完成）
- 密码仍为明文

---

### 2026-03-01 — 初始部署 + React SPA 迁移 + 认证修复

#### Goal
将项目从 Flask/Jinja2 模板渲染迁移至 React 18 + Vite SPA；修复认证层的重大安全漏洞；部署至 AWS EC2 并完成 HTTPS 配置；建立基础文档体系。

#### Trigger / Context
项目原始版本使用 Flask 直接渲染 HTML 模板，难以扩展复杂的前端交互。计划将平台升级为前后端分离架构，为后续功能开发打好基础。

#### Problem & Root Cause

**严重安全漏洞**：`authenticate_user` 函数存在硬编码后门：`password in ['horizon', 'yyf']` — 任何用户用这两个密码均可登录。

**用户查找 bug**：所有用户管理方法使用 `users.json` 的 dict key 查找用户，而非 `username` 字段。当两者不一致时（存量数据）登录失败。

**部署后登录失效**：生产环境登录后 cookie 不生效。根因：Nginx 反代后 Flask 看到的是 HTTP 请求，`SESSION_COOKIE_SECURE=True` 拒绝在 HTTP 上写 cookie。

#### Solution

**React SPA 迁移**
- 删除 `Template/`（Jinja2 模板）和 `Static/`（旧静态文件）目录
- Flask 改为纯 API-only，所有路由统一在 `/api/*` 下
- React 18 + Vite 构建，产出 `frontend/dist/`
- Flask `serve_react()` catch-all 路由伺服 `index.html`

**认证修复**
- 移除硬编码后门 `password in ['horizon', 'yyf']`
- 全局统一用 `_find_user()` helper 按 `username` 字段查找，修复 `auth_controller.py` 和 `memos_controller.py`
- 修复 `users.json` 存量数据中 key/username 不一致问题

**部署配置**
- `ProxyFix(x_for=1, x_proto=1, x_host=1)` 信任 Nginx 代理头
- `SESSION_COOKIE_SECURE` 在 `LOCAL_DEV=1` 时关闭
- Nginx 配置：`/etc/nginx/conf.d/horizonyhj.com.conf`（反代到 Gunicorn :8000）
- systemd service：`/etc/systemd/system/horisation.service`
- SSL：Let's Encrypt + Cloudflare Full 模式
- Python venv：`/home/ec2-user/venv/`（后续升级至 3.11）

**功能清理**
- 移除 `/limit` 路由和 `limit.html`（废弃功能）
- 移除 `last_login` 字段（导致 `users.json` 频繁 git 冲突）

**React 初版页面**
Login、Home、CSV Workspace、Hormemo、Profile、AdminUsers、Under Development、Gomoku（本地双人）

#### Changed Files
- `app.py` — 重构为 API-only + SPA catch-all；加 ProxyFix 和 cookie 安全配置
- `Backend/Controller/auth_controller.py` — 移除后门；修复 `_find_user()` 调用
- `Backend/Controller/memos_controller.py` — 修复 `_find_user()` 调用
- `Template/` — 删除目录
- `Static/` — 删除目录
- `frontend/` — 新建，React 18 + Vite 项目结构
- `frontend/src/App.jsx` — 路由、AuthContext、PrivateRoute
- `frontend/src/api.js` — fetch wrapper（`credentials: include`）
- `frontend/src/pages/` — 初版所有页面组件
- `requirements.txt` — 新建（flask, pandas, numpy, openpyxl, xlrd, pyarrow, gunicorn）
- `Doc/project_intro.md`, `Doc/server.md`, `Doc/log.md`, `Doc/data_storage.md` — 新建文档

#### Result
- 生产环境 HTTPS 正常运行，登录 cookie 正常设置
- 安全漏洞修复，用户查找逻辑统一
- React SPA 全面上线，前后端分离架构就绪
- 基础文档建立

#### Testing
- 生产环境用正常账号登录验证 cookie 正常写入，session 跨页面保持
- 确认硬编码后门密码（`horizon`/`yyf`）不再绕过认证
- 用存量用户数据验证 `_find_user()` 修复后所有账号可正常登录
- 访问 `/login` 后重定向至 `/home`；直接访问 `/home` 未登录时重定向至 `/login`

#### Lessons Learned
- **Symptom**: 生产登录失败，本地正常
- **Root Cause**: 反向代理改变了请求协议头，Flask 无法正确判断 HTTPS
- **Reusable Solution**: 所有部署在反代后的 Flask 应用都应加 `ProxyFix`；本地用环境变量关闭 `SESSION_COOKIE_SECURE`，避免掩盖问题

- **Symptom**: 代码中存在硬编码测试凭证
- **Root Cause**: 开发阶段为方便调试遗留的临时代码未及时清理
- **Reusable Solution**: 代码 review 时专项检查硬编码凭证；生产部署前运行 `grep -r "password in \[" .` 类扫描

#### Remaining Issues / Next Step
- 二手市集和留言板（下一个迭代完成）
- `users.json` 并发安全问题（2026-03-06 通过迁移 SQLite 解决）
- 密码明文存储

---

## Deploy Checklist
```bash
# Local — push changes
git add -A && git commit -m "..." && git push

# Server — one command
bash ~/deploy.sh
```
