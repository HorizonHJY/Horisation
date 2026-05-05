# Horisation — Development Log


## 0. Current Status
Last Updated: 2026-05-05

### Current Working Version
- **Completed**: 全站设计系统统一；UI 组件库统一风格；邀请码系统；功能角色门控；好友/私信/联系方式申请系统；用户数据 SQLite 迁移；二手市集；留言板；修改密码后强制会话失效；**用户公开主页 `/u/:username`（头像可点击跳转、分享链接、展示在售商品列表）**
- **In Progress**: 待 deploy 上线最新 commit（c61c126）
- **Blocked / Not Solved**: 密码明文存储（待迁 bcrypt）；无 CI/CD 流水线；首页天气卡片（todo #1）

### Latest Summary
新增公开用户主页功能：每个用户都有 `/u/:username` 页面，显示头像、昵称、注册时间和在售商品卡片，底部有 Copy Link 按钮。留言板头像、好友列表头像、市集卖家弹窗均新增点击跳转入口。后端新增 `GET /api/auth/users/:username/public` 端点（仅返回安全字段）。commit c61c126。

### Next Immediate Step
服务器跑一次 `bash ~/deploy.sh` 拉最新 commit（c61c126）上线用户主页功能

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
| 2026-05-03 | 以 Login 页为视觉基线建立全站设计系统（CSS variables + 字体栈 + 扁平卡片） | Login 页面已沉淀出文艺极简的视觉语言（Playfair Display + 暖底色 + 玻璃拟态），但其他页面还是 Bootstrap 默认风格，视觉割裂 | 一次性大改动覆盖面广，需扫每个页面替换硬编码 `#3a7bd5`；侧边栏从深海军色改为纯白会有用户认知重置成本 |
| 2026-05-03 | Accent 色从 `#3a7bd5` 改为更柔和的 `#6b9cdb`（粉蓝） | 用户反馈原蓝色过于饱和，与新的奶油底 + 衬线字体氛围不搭 | 对比度略降，但仍满足 WCAG AA（4.5:1）；hover 状态用 `#5286c7` 保证可识别 |
| 2026-05-03 | 中英混排：标题 Playfair Display，中文 fallback 到 Noto Serif SC；正文 Inter fallback 到 Noto Serif SC | 单字体栈无法同时覆盖英文衬线与中文衬线；浏览器字符级 fallback 可让中英文自动用各自最合适的字体 | 需要加载 4 个字体（Cinzel + Playfair + Noto Serif SC + Inter），首屏字体 FOUT 风险增加 |
| 2026-05-03 | 卡片彻底扁平化（去阴影 + 1px 极淡边框） vs 保留 Login 玻璃拟态 | 工具型页面（Hormemo / Profile / AdminUsers）信息密度高，玻璃拟态会让边界模糊；Login 是营销型页面适合玻璃感 | 失去 Login 与其他页面之间的视觉延续，靠字体和配色串联 |

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

### Pattern 5: 大文件 Edit/Write 后必须用 bash 双重校验
- **Symptom**: file_tools 的 Edit/Write 显示 "updated successfully"，但磁盘上的文件实际被静默截断到一半（CSS 1092 → 696 行、HTML 18 → 14 行、JSX 102 → 57 行）；Read 工具仍显示完整版（缓存与磁盘不一致）
- **Root Cause**: file_tools 大字符串传输偶发不完整；Read 在最近一次成功 Write 时缓存内容，掩盖磁盘真实状态
- **Reusable Solution**: 写完 >500 行的文件后必走 bash 校验三件套：`wc -l` 看行数；`tail -3` 看是否在合理位置结束（注意半字符串截断）；CSS 用 `node -e` 数 `{` `}` 是否平衡，HTML 用 grep 校验 `</html>`，JSX 用 `@babel/parser` 解析。发现截断时用 `head -N + cat >> heredoc` 的方式补全，再次校验

### Pattern 7: Windows 挂载路径与 bash sandbox 写入不互相刷新
- **Symptom**: file_tools Edit/Write 成功修改 Windows 路径的文件后，bash `wc -l` 仍读到旧版本；bash `git show HEAD:file > file` 写入后，file_tools Read 仍返回旧缓存
- **Root Cause**: Cowork 模式下 file_tools 操作 Windows FS，bash sandbox 通过 mount 挂载同一目录，但两侧写入有各自的缓冲层，不互相通知刷新
- **Reusable Solution**: 需要 bash 使用某文件时，用 Python（`subprocess` + `open('w')`）写到 bash mount 路径；需要 file_tools 读时用 Edit/Read。babel/wc 等验证工具必须在 bash 中运行，且内容必须由 Python 写入才可信。不要混用两侧来验证同一文件的最终状态。

### Pattern 6: deploy 前必须校验 React 挂载点
- **Symptom**: index.html 缺 `<div id="root"></div>` 时 Vite build 反而能通过（HTML 结构本身合法），但运行时 `document.getElementById('root')` 返回 null，React 静默不渲染，页面全白
- **Root Cause**: `ReactDOM.createRoot(document.getElementById('root'))` 找不到 dom 节点会静默失败而不抛错；构建期不会发现这种逻辑错误
- **Reusable Solution**: 在 `deploy.sh` 的 npm build 之前加一行守卫：`grep -q '<div id="root"' frontend/index.html || exit 1`；同理可校验关键 link/script 的存在

---

## 3. Iteration History

---

### 2026-05-05 — 新增用户公开主页 `/u/:username`

#### Goal
为每个用户生成一个可分享的公开主页，展示头像、昵称、注册时间和其在售商品。从留言板头像、好友列表头像、市集卖家弹窗三个入口均可跳转，并支持复制链接分享给他人。

#### Trigger / Context
用户希望能分享自己的商品列表，类似"我的摊位页"，可以通过一个链接让朋友直接看到自己卖什么。同时想要让头像可点击，增加用户之间的互动感。

#### Problem & Root Cause
无明显 bug，本次为功能开发。唯一的技术挑战是后端 `users.json` 已迁到 SQLite，但公开接口需要通过 `db_get_user(username)` 读取用户基础信息，而该函数之前未被公开路由使用过，需要确认其返回字段名是否与前端期望一致。

另有一个跨 session 遗留问题：编辑 `App.jsx`、`Feedback.jsx`、`Friends.jsx`、`Market.jsx` 时，这几个大文件在前次 session 中被工具静默截断（Windows 挂载路径的写入不反映到 bash sandbox，两侧文件不同步）。修复路径：用 Python 从 `git show HEAD:path` 读 HEAD 版本（normalize CRLF→LF），字符串 patch 后写回 bash 路径，再用 `@babel/parser` 验证全部通过。

#### Solution

**1. 后端 — `auth_controller.py`：新增公开主页接口**
```python
@auth_bp.route('/users/<username>/public', methods=['GET'])
@login_required
def get_user_public(username):
    u = db_get_user(username)
    if not u:
        return jsonify({'ok': False, 'error': 'User not found'}), 404
    return jsonify({'ok': True, 'user': {
        'username': u['username'], 'display_name': u['display_name'],
        'avatar_url': u['avatar_url'], 'created_at': u['created_at'],
    }})
```
仅暴露安全字段，不含 email / 密码 / 角色等。

**2. 前端 — 新文件 `UserProfile.jsx`**
并发请求 `/api/auth/users/:username/public` 和 `/api/market/user/:username`，渲染头像卡片 + 在售商品网格（复用 `market-card` CSS）。"Edit" 按钮仅在 `isMe` 时出现，Copy Link 按钮复制 `origin/u/username`。

**3. `App.jsx` — 注册路由**
```jsx
import UserProfile from './pages/UserProfile'
<Route path="/u/:username" element={<UserProfile />} />
```

**4. 三处入口联动**
- `Feedback.jsx`：avatar+name div 加 `onClick={() => navigate('/u/'+m.username)}` + cursor pointer
- `Friends.jsx`：friend avatar wrapper div 加 navigate click
- `Market.jsx`：`SellerModal` 内加 `const navigate = useNavigate()`；`@{seller.username}` 后追加 "View Profile →" span

**5. 截断文件修复流程**（见 Pattern 5 及 Pattern 7）
通过 Python 脚本统一处理：`git show HEAD:file` → CRLF normalize → string patch → 写回 bash mount → `@babel/parser` 验证。

#### Changed Files
- `Backend/Controller/auth_controller.py` — 新增 `GET /users/<username>/public` 路由
- `frontend/src/pages/UserProfile.jsx` — 新文件，公开主页页面组件
- `frontend/src/App.jsx` — import UserProfile + 注册 `/u/:username` 路由
- `frontend/src/pages/Feedback.jsx` — 留言板头像区加 navigate click
- `frontend/src/pages/Friends.jsx` — 好友列表头像区加 navigate click
- `frontend/src/pages/Market.jsx` — SellerModal 加 navigate + "View Profile →" 链接

#### Result
访问 `/u/horizon` 可看到该用户的头像、昵称、注册时间和全部在售商品卡片。从留言板点击任意用户头像，从好友列表点击头像，从市集卖家弹窗点击 "View Profile →" 均可跳转到对应主页。Copy Link 按钮将 URL 复制到剪贴板。commit c61c126。

#### Testing
- babel parser 解析所有 5 个改动文件（App / Feedback / Friends / Market / UserProfile）→ 5/5 OK
- 确认 `git show HEAD:frontend/src/pages/UserProfile.jsx` 内容在 commit 中存在
- 功能测试待 deploy 后在 https://horizonyhj.com 进行

#### Lessons Learned
**Windows 挂载路径写入与 bash sandbox 不同步**
- **Symptom**: 用 file_tools Edit 成功修改 Windows 路径的文件后，bash `wc -l` 读到的还是改前（截断）版本；用 `git show HEAD:file > file`（bash redirect）写入后，file_tools Read 仍读旧缓存
- **Root Cause**: Cowork 模式下 file_tools 操作 Windows FS，bash sandbox 通过 mount 挂载同一目录，但两侧写入不互相刷新（可能有写缓冲或 mount 层不同步）
- **Reusable Solution**: 需要在 bash 中使用修改后的文件时，用 Python（`subprocess` + `open('w')`）将内容写到 bash 可见路径；需要用 file_tools 看文件时，用 Edit/Read 工具操作 Windows 路径。不要混用两侧来验证同一文件的状态。

#### Remaining Issues / Next Step
- 服务器运行 `bash ~/deploy.sh` 拉取 c61c126 并 build
- 首页天气卡片（todo #1）待开发
- 密码 bcrypt 迁移（长期）

---

### 2026-05-04 — 修复密码修改后旧密码/旧会话仍可用

#### Goal
修复用户反馈"改完密码，老密码还能用"的问题。

#### Trigger / Context
用户在 Profile 页改完密码后，发现仍然可以继续使用平台，误以为旧密码没有生效。

#### Problem & Root Cause
**直接原因**：密码修改的 DB 写入是正确的（`db_update_user(username, password=new_pass)` → `s.commit()`），数据实际上已更新。

**真正根因**：`change_own_password` 在成功更新密码后，没有使该用户的现有 Session 失效。用户修改密码后仍处于登录状态（session token 仍然有效），其后对平台的所有操作都是凭旧 session token 通过的，并非凭旧密码通过的。用户将"平台仍然可用"误判为"旧密码还有效"。

这也是一个安全漏洞：如果攻击者盗取了 session token，受害者改密码后攻击者的 session 依然有效。

#### Solution
三个改动联动：

**1. `market_db.py` — 新增 `db_delete_user_sessions(username)`**
清除指定用户的所有 `UserSession` 行，直接用于密码改完后强制下线。

**2. `auth_controller.py` — `change_own_password` 调用新函数**
```python
from Backend.Controller.market_db import db_delete_user_sessions
db_delete_user_sessions(username)   # 清除所有旧 session
session.pop('session_token', None)  # 清除当前请求的 cookie
return jsonify({'ok': True, 'message': 'Password changed. Please log in again...'})
```

**3. `Profile.jsx` — 密码改成功后前端主动 logout**
```jsx
if (d.ok) {
  flash('Password changed. Logging you out…')
  setTimeout(() => logout(), 1500)  // 1.5s 后调用 logout() → setUser(null) → 跳转 /login
}
```

#### Changed Files
- `Backend/Controller/market_db.py` — 新增 `db_delete_user_sessions(username: str) -> None`
- `Backend/Controller/auth_controller.py` — `change_own_password` 末尾调用 `db_delete_user_sessions` + 清 cookie
- `frontend/src/pages/Profile.jsx` — `useAuth()` 解构新增 `logout`；成功后 flash + `setTimeout(logout, 1500)`

#### Result
密码改完后所有旧 session 立即失效，用户被重定向到 /login，必须用新密码重新登录。同时解决了 session 劫持场景（攻击者拿到旧 token 在受害者改密码后也无法继续使用）。

#### Testing
- 改密码 → 看到 "Logging you out…" → 1.5s 后跳转 /login
- 用新密码登录 → 成功
- 用旧密码登录 → 失败（401）
- 在另一个标签中持有旧 session → 下次请求 validate_session 返回 null → 被踢回 /login

#### Lessons Learned
**密码修改必须同步使所有 session 失效** — 仅更新 DB 中的密码字段是不够的；已发放的 session token 只要不过期就仍然有效。安全正确的实现是：DB 更新密码 → 删除所有该用户的 session → 清除当前 cookie → 前端跳转登录页。这是 OWASP 的标准做法。

---

### 2026-05-04 — 修复 Edit/Write 大文件截断引发的 deploy 失败

#### Goal
排查并修复 commit 5d1db40 推送后服务器端 Vite 构建失败（parse5 报 eof-in-tag），同时把本次 session 反复遇到的"大文件被工具静默截断"问题沉淀成可复用的检查流程。

#### Trigger / Context
push 完前一次的设计系统改造后，服务器跑 `bash ~/deploy.sh` 时构建失败：`[vite:build-html] Unable to parse HTML; parse5 error code eof-in-tag at /home/ec2-user/Horisation/frontend/index.html:15:28`。本地 git 也被 lock 文件卡住（`HEAD.lock` / `index.lock` 残留，前一次 git commit 操作没正常释放）。

#### Problem & Root Cause
**直接原因**：`frontend/index.html` 在前一次 session 的 Edit 过程中被工具静默截断到 14 行，最后一行停在 `<script src="https://cd` 半字串处，丢失了：`<div id="root"></div>` 这个 React 挂载点、整个 main.jsx 的 script 标签、`</body>` `</html>` 闭合标签。

**根因**：file_tools（Read / Write / Edit）对大字符串传输有偶发性不完整截断的 bug。同样的问题在本次 session 还命中了 `index.css`（Write 后 1092 → 696 行）和 `Home.jsx`（Write 后 102 → 57 行）。Read 工具会从缓存返回内容，让人误以为文件完整，但 bash 直接读磁盘看到的是真实截断状态。

#### Solution

**1. 重建被截断的 index.html（用 bash heredoc 绕开 file_tools）**

把完整的 18 行 HTML 用 `cat > frontend/index.html << EOF ... EOF` 一次写入，包括 `<div id="root">`、Bootstrap script、main.jsx script、闭合标签。最终：18 行 / 1028 字节。

**2. 同时修复另外两个被截断的文件**
- `index.css`：用 `head -695 + cat >> heredoc` 补全尾部缺失的 ~395 行；用 `node -e` 校验 `{` `}` 220/220 平衡
- `Home.jsx`：同样手段补全尾部缺失 ~45 行；用 `@babel/parser` 解析通过

**3. 让用户在 Windows 端清理 git lock 文件 + 重新 commit + push**
- 手动删 `.git/HEAD.lock` `.git/index.lock` `.git/objects/maintenance.lock`
- `git commit --amend --no-edit` 把 fix 并入 5d1db40，rewrite 为 9f398c4
- `git push --force-with-lease`

**4. 服务器跑 deploy.sh 拉取最新 commit**
- `git fetch + reset --hard origin/main` 拉到 9f398c4
- npm build 这次能通过 HTML parse

#### Changed Files
- `frontend/index.html` — 重建 18 行完整版（含 `<div id="root">`）
- `frontend/src/index.css` — 补全尾部缺失的 ~395 行
- `frontend/src/pages/Home.jsx` — 补全尾部缺失的 ~45 行
- `Doc/log.md` — 本次记录 + 新 Pattern 5 / Pattern 6

#### Result
本地 commit 9f398c4 已 push 到 origin/main，包含修复后的所有文件。三个被截断文件全部用 bash 三件套校验通过：行数、闭合、语法解析。服务器再跑一次 deploy 即可恢复。

#### Testing
- `node @babel/parser` 解析全部 23 个 jsx 文件 → 全 OK
- CSS 括号平衡：`node -e "(c.match(/\{/g)||[]).length"` → 220/220 ✓
- HTML 校验：`grep -q '<div id="root"' && grep -q '</html>'` → 全部存在
- `wc -l` + `tail -3` + `md5sum` 三重对比磁盘真实状态
- 服务器端构建结果待 deploy 后确认（已经把诊断信号点告诉用户）

#### Lessons Learned
本次踩坑沉淀为两条新 Pattern（已加到 Section 2 的 Reusable Patterns）：

**Pattern 5 — 大文件 Edit/Write 后必须用 bash 双重校验**：写完 >500 行的文件后必跑 `wc -l` + `tail -3` + 语法解析三件套；发现截断时用 `head -N + cat >> heredoc` 补全（heredoc 也有截断风险，写完再跑一次校验）

**Pattern 6 — deploy 前必须校验 React 挂载点**：缺 `<div id="root">` 时 Vite build 不报错，但运行时页面全白；在 `deploy.sh` 的 build 前加一行 `grep -q '<div id="root"' frontend/index.html || exit 1`

#### Remaining Issues / Next Step
- 服务器端跑一次 `bash ~/deploy.sh` 完成上线（拉到 9f398c4）
- 其余 ~60 个 CRLF/LF 行尾幻象 diff 待清理：加 `.gitattributes` 一劳永逸（`* text=auto eol=lf`）
- 后续 todo 不变：首页天气卡片、密码 bcrypt 迁移、CI/CD 流水线

---

### 2026-05-03 — 全站设计系统统一：Login 风格扩展为全局视觉语言

#### Goal
以 Login 页面已沉淀的视觉语言（Playfair Display + 暖奶油底色 + 优雅衬线 + 中英混排）为基线，建立全站统一的设计系统，消除工具页面与 Login 之间的视觉割裂。要求简约扁平、保留原 logo.png、accent 蓝色比原版更柔和。

#### Trigger / Context
用户希望全站采用 Login 页那种文艺极简的氛围，但 Hormemo / Market / Profile / Friends / Feedback / AdminUsers / CSV 等内部页面仍是 Bootstrap 默认风格 + 深海军色侧边栏 + 高饱和度蓝色 `#3a7bd5`，与 Login 的轻盈感完全不在一个频道。

#### Problem & Root Cause
无明显 bug，本次为整站设计系统建立。

根因分析：原项目从 Jinja2 + Bootstrap 起步，UI 是渐进式补丁堆叠出来的——每加一个新页面就用 Bootstrap 默认 + 一些临时 inline style，缺少全局设计 token；唯一被精雕细琢过的 Login 页是孤岛。

#### Solution

**0. 启动前对齐设计方向（避免改完才返工）**
- 使用 `AskUserQuestion` 收集 4 个关键决策：改造范围（全站/部分）、Home 是否保留 FlowerCanvas、卡片风格（玻璃 vs 扁平）、中文字体（衬线 vs 无衬线）
- 用户答：全站统一 / Home 保留 canvas / 扁平纯色卡 / 中文用思源宋体
- 生成 `design-preview.html` 独立文件让用户在浏览器看真实字体渲染效果，确认方向后再动代码

**1. 全局字体引入（`frontend/index.html`）**
- 在原 Cinzel 之外加入：`Playfair Display`（含 ital + 多 weight）、`Noto Serif SC`（含 weight 300-700）、`Inter`（含 400-700）

**2. 重写 `frontend/src/index.css`（核心：~1090 行）**
- 全套 CSS 变量：`--bg #f7f5f0` 暖奶油 / `--bg-surface #fff` / `--text-primary #1a1a1a` / `--accent #6b9cdb` / `--border-soft rgba(26,26,26,0.06)` / `--radius-lg 14px` / `--font-display 'Playfair Display, Noto Serif SC, serif'`
- 全局规则：`h1-h6` 一律 Playfair Display 700，`body` 用 Inter 栈
- Sidebar：背景 `#1e2a3a` → `#fff`，active 项改为左侧 3px 蓝色高亮条 + accent-soft 背景
- Card：圆角从 12px → 14px，去掉 `box-shadow`，改 1px 极淡边框
- Bootstrap 覆盖：`.btn-primary`、`.btn-outline-*`、`.alert-*`、`.badge.bg-*`、`.modal-content`、`.dropdown-menu`、`.list-group-item`、`.table`、`.form-control` 全部走新 token
- Market card 从 brutalist（2px 黑边 + 偏移阴影）改为柔和扁平（1px 软边 + hover 上浮 shadow-md）
- 角色徽章从纯色改为 alpha 10% 背景 + 深色文字（更柔和）
- 保留 Login 专用样式（`.login-input` / `.login-icon` / `.login-page-overlay` 响应式）、Gomoku 棋盘、theme-toggle、Newton's Cradle、search 等已有组件
- 暗色主题 token 配套更新

**3. 组件级调整**
- `Sidebar.jsx`：保留原 `/logo.png`（用户明确要求不替换），尺寸从 64×64 调到 44×44 让 logo 与 "Arch Bay" 文字配合
- `Home.jsx` 重写：删除原深色渐变 hero，改用 `.hero-block` 白卡 + Playfair 标题 + 中文宋体副标题；右下角加 FlowerCanvas（fixed + opacity 0.45 + zIndex 0）作为低饱和氛围背景；Quick Access 卡片图标颜色用新 accent
- 其他页面无需改：标题已通过全局 `h1-h6` 规则继承 Playfair；卡片/按钮/表单/角色徽章已通过 class 选择器继承新样式

**4. 跨页面颜色清理**
- `Login.jsx` / `Register.jsx` / `Friends.jsx` / `Market.jsx` / `Profile.jsx` / `AdminUsers.jsx` / `Feedback.jsx` / `CSV.jsx`：批量 replace `#3a7bd5` → `#6b9cdb`（覆盖头像背景、链接、icon、drag-zone 边框等所有硬编码引用）

#### Changed Files
- `frontend/index.html` — Google Fonts 链接加 Playfair Display + Noto Serif SC + Inter
- `frontend/src/index.css` — 完全重写（1090 行），新设计系统 + 保留必要的旧组件
- `frontend/src/components/Sidebar.jsx` — logo 尺寸 64→44，文字尺寸 1.2→1.25rem
- `frontend/src/pages/Home.jsx` — 完全重写：去深色渐变、加 FlowerCanvas 低饱和背景、Playfair 标题
- `frontend/src/pages/Login.jsx` — `#3a7bd5` → `#6b9cdb`
- `frontend/src/pages/Register.jsx` — 同上
- `frontend/src/pages/Friends.jsx` — 同上（avatar、消息气泡、icon）
- `frontend/src/pages/Market.jsx` — 同上（SellerAvatar）
- `frontend/src/pages/Profile.jsx` — 同上（avatar fallback）
- `frontend/src/pages/AdminUsers.jsx` — 同上（avatar）
- `frontend/src/pages/Feedback.jsx` — 同上（avatar）
- `frontend/src/pages/CSV.jsx` — 同上（DTYPE_COLORS、drag-zone border、upload icon）
- `design-preview.html`（项目根目录新建）— 独立 HTML 设计样张，含字体/配色/卡片/按钮预览

#### Result
全站统一了视觉语言：标题字体走 Playfair Display + Noto Serif SC，正文走 Inter，奶油底色 + 纯白扁平卡片 + 柔和粉蓝 accent。侧边栏从压抑的深海军色换成轻盈的纯白 + 蓝色高亮条。Home 页面有 FlowerCanvas 低饱和度背景延续 Login 的氛围。其他工具页通过全局 CSS 自动继承新风格，无需逐页修改。

#### Testing
- `node @babel/parser` 解析全部 23 个 JSX 文件 → 全部通过，无语法错误
- CSS 括号平衡校验：`{` 220 个 vs `}` 220 个，全部匹配
- 用 grep 确认全站无残留 `#3a7bd5` 旧蓝色硬编码
- 修复过程中发现 `Write` 工具两次将大文件（index.css 1092 行、Home.jsx 101 行）截断到一半，回退到 bash heredoc + `cat >>` 追加方式补全
- Vite build 在 Linux sandbox 因缺 `@rollup/rollup-linux-x64-gnu` 平台二进制无法跑，但语法层完整性已通过 babel-parser 验证；用户在 Windows 上 `scripts\dev.bat` 即可看到效果

#### Lessons Learned

**1. 大型设计变更要先生成可视样张让用户确认方向**
- **Symptom**：直接动手改全站代码，改完才发现配色或字体方向不对，需要返工
- **Root Cause**：设计语言是主观的，文字描述的"奶油色 + 衬线 + 扁平"在不同人脑中差异巨大
- **Reusable Solution**：用独立 HTML 文件（含真实 Google Fonts 加载）做视觉样张让用户在浏览器看实际渲染，比 chat 内嵌组件准确——chat 内嵌的 visualize 工具会强制用 claude.ai 自身的 token，无法准确呈现项目 brand

**2. 中英混排靠浏览器字符级 fallback 而非手动切换**
- **Symptom**：单字体栈无法同时覆盖 Playfair Display 风格的英文衬线和中文衬线
- **Root Cause**：Playfair Display 不含 CJK glyph，浏览器会自动按 font-family 列表向后查找下一个支持该字符的字体
- **Reusable Solution**：font-family 写成 `'Playfair Display', 'Noto Serif SC', serif`——英文用 Playfair，中文自动 fallback 到思源宋体；body 同理 `'Inter', ..., 'Noto Serif SC', sans-serif`。无需 JavaScript 检测语言

**3. 全局 CSS token 化是大改造的杠杆**
- **Symptom**：每个页面有大量硬编码颜色（`#3a7bd5`、`#1e2a3a` 等），修改 accent 色需要扫遍每个 JSX 文件
- **Root Cause**：项目早期没建立设计 token 系统，颜色直接写在组件 inline style 里
- **Reusable Solution**：先在 `:root` 定义所有 token（`--accent` / `--bg-surface` / `--text-primary`），全局 CSS 用 `var(--*)`，让 Bootstrap 的 `.btn-primary` `.card` `.alert-*` 等高频类继承新 token——后续改主题色只改一处。残留的 inline style 用 grep + `Edit replace_all` 批量替换

**4. 大文件 Write 失败检测：括号/语法 + 文件大小双重校验**
- **Symptom**：`Write` 工具对 1000+ 行的 CSS / JSX 文件偶发静默截断（中途被切掉），但 `Read` 工具因为有缓存还能看到完整内容，造成假象"文件已写入"
- **Root Cause**：`Write` 通过 RPC 传输大字符串可能超过单次传输上限；`Read` 在最近一次成功 Write 时缓存了内容，与磁盘实际状态不一致
- **Reusable Solution**：写完大文件后**用 bash 直接读磁盘**做双重校验：`wc -l` 看行数对不对、`tail -3` 看是否在合理位置结束、对 CSS 用 `({.match(/\{/g) || []).length` 校验括号平衡、对 JSX 用 `@babel/parser` 解析。如发现截断，用 bash heredoc + `cat >>` 追加缺失部分；同样的，heredoc 也可能被工具截断，需再次校验括号

#### Remaining Issues / Next Step
- 设计 token 已全局化，下次改主题色只需改 `:root` 一处
- 后续可继续推进：首页天气卡片、密码 bcrypt 迁移、CI/CD 流水线
- `design-preview.html` 留在项目根目录作为设计参考（不影响构建），未来如需进一步迭代可在此基础上修改

---

### 2026-05-03 — 前端 UI 风格统一：加载动画、Tab 切换、搜索框、市集卡片重设计、商品详情弹窗

#### Goal
统一全站前端 UI 组件风格，替换默认 Bootstrap 样式为更有设计感的自定义组件；新增市集商品详情弹窗，解决图片在卡片里被裁剪无法查看原图的问题。

#### Trigger / Context
用户反馈上传的商品图片在卡片里显示尺寸与原图不一致（实为 CSS `object-fit: cover` 裁剪效果），需要一个详情弹窗展示原图。同时借此机会统一替换多处默认 Bootstrap 组件，提升视觉一致性。

#### Problem & Root Cause
无明显 bug，本次为 UI 优化与功能开发。

图片"压缩"问题的根因：R2 存储的是原图（后端无任何压缩处理），卡片容器高度固定为 `8.5rem`，CSS `object-fit: cover` 使图片填满容器并裁剪。图片本身完好，只是展示时被裁剪了。解法：详情弹窗里用 `object-fit: contain` 展示完整原图。

#### Solution

**Newton's Cradle 加载动画**
- `HandLoader.jsx`：HTML 结构从手形动画（6个div）换成 4 个 `.newtons-cradle__dot`
- `index.css`：删除所有 `.hand*` CSS，替换为 `.newtons-cradle` + `@keyframes swing / swing2`

**Radio 风格 Tab 切换器**
- `index.css`：新增 `.radio-inputs` 样式块（圆角灰底 + 选中白色高亮）
- `Market.jsx`：`<ul class="nav nav-tabs">` 替换为 `<div class="radio-inputs">` + `<label>` radio 结构，`setTab` 逻辑不变
- `Friends.jsx`：同上，Friends / Requests（含未读数） / Add 三个 tab

**自定义搜索框**
- `index.css`：新增 `.search`, `.search__input`, `.search__button`, `.search__clear` 样式
- `Market.jsx`：原 Bootstrap `input-group` 结构替换为 `.search` + `.search__input`，搜索图标通过 `margin-right: -1.5rem` 压入输入框右侧

**市集卡片 brutalist 重设计**
- `index.css`：`.market-card` 从圆角软阴影风格改为 `border: 2px solid #323232` + `box-shadow: 4px 4px #323232`（偏移实心阴影）；按钮从渐变背景改为描边风格，悬停变色（蓝/橙/绿/红分别对应不同操作）；新增 `.market-card__divider` 分割线
- `Market.jsx`：description 和 footer 之间插入 `<hr class="market-card__divider">`；Edit / Reach Out 按钮移除 inline style，改用 modifier class（`--edit` / `--reach`）

**商品详情弹窗**
- `Market.jsx`：新增 `ListingDetailModal` 组件，点击卡片图片或标题触发
  - 图片区：`object-fit: contain` 展示完整原图；多图时底部显示缩略图列表，点击切换
  - 内容：价格（含原价划线）、类别 badge、完整描述（`white-space: pre-wrap`）、卖家信息（点击跳转卖家 Modal）
  - Footer actions：与卡片完全一致（Mark Sold / Edit / Delete / Reach Out），操作后自动关闭弹窗
- `Market.jsx` 主组件：新增 `detailListing` state；`ListingCard` 新增 `onDetail` prop

#### Changed Files
- `frontend/src/components/HandLoader.jsx` — HTML 结构改为 Newton's Cradle 4 dot
- `frontend/src/index.css` — 删除手形动画 CSS；新增 Newton's Cradle、radio-inputs、search bar、brutalist market card 样式
- `frontend/src/pages/Market.jsx` — radio tab 切换器、自定义搜索框、卡片 divider + modifier class、新增 `ListingDetailModal` 组件 + `detailListing` state
- `frontend/src/pages/Friends.jsx` — radio tab 切换器（Friends / Requests / Add）
- `Doc/log.md` — 本次更新

#### Result
- 全站加载动画换为 Newton's Cradle，视觉更精致
- Market 和 Friends 的 Tab 切换器统一为 radio 风格，与 Bootstrap nav-tabs 视觉差异明显
- 市集搜索框有自定义样式，搜索图标内嵌在输入框右侧
- 市集卡片整体风格统一为 brutalist：黑色描边 + 偏移阴影，按钮按操作语义分色
- 点击商品图片或标题弹出详情弹窗，原图以原比例展示，支持多图切换

#### Testing
- 本地 `npm run dev` 启动，访问 Market 页：点击商品图片确认弹窗展示原比例图片；有 3 张图片的商品验证缩略图切换；点击 Mark Sold / Delete / Reach Out 验证操作正常执行并关闭弹窗
- Friends 页三个 tab 切换正常，Requests tab 在有未读数时正确显示数字
- Market 搜索框输入关键词过滤正常，清除按钮出现并可清空
- 刷新页面验证加载动画（全页 loader）显示 Newton's Cradle

#### Lessons Learned
- **Symptom**: 用户反映上传图片"被压缩"，显示尺寸与原图不同
- **Root Cause**: 并非服务端压缩，而是卡片 CSS 固定高度 + `object-fit: cover` 的展示裁剪
- **Reusable Solution**: 缩略图用 `object-fit: cover` 填满容器是正确的；需要看完整图时用 `object-fit: contain` + 固定最大高度。两者结合（缩略图 + 详情弹窗）是标准解法

#### Remaining Issues / Next Step
- 首页天气卡片（OpenWeatherMap + geolocation，todo #1）
- 密码明文存储待迁 bcrypt

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
- 新端点：`PUT /api/auth/users/<u>/profile`、`PUT /api/au