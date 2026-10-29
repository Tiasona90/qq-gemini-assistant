# 基于 Gemini API 的私人 QQ 智能助手 (QQ Gemini Assistant)

轻量、安全、合规的单机个人 QQ 智能问答与生活知识助手后端。基于 Google 官方最新的 `google-genai` Python SDK、SQLite WAL 存储和 OneBot 11 协议标准构建。

---

## 1. 技术架构

整体采用单机分层解耦架构：

```
[NapCat / OneBot 11 (QQ)] 
       ↕ (WebSocket 本地反向连接 / 免公网IP)
[QQ Adapter (OneBot 11)]
       ↕
[Chat Manager (消息去重 / 白名单 / 队列与并发控制 / 会话隔离 / 命令解析)]
       ↕
[SQLite 数据库 (WAL 模式，存储历史上下文与用量统计)]
       ↕
[Gemini Client (google-genai 官方 SDK / 指数退避重试 / Google Search Grounding)]
```

### 核心模块职责
- **QQ Adapter (`qq_adapter/`)**：负责通过 OneBot 11 反向 WebSocket 接收私聊消息、验证访问令牌、向 QQ 发送回复，并在无 QQ 环境下提供命令行模拟终端 (`ConsoleAdapter`)。
- **Chat Manager (`chat_manager.py`)**：核心调度层，处理用户白名单校验、消息去重 (Deduplication)、单用户 FIFO 排队锁、全局 API 并发限制 (默认最大 2)、标准/模糊指令处理、健康/法律风险防范提示及纯文本回复排版。
- **Database (`db.py`)**：基于 `aiosqlite` 的本地异步数据库，持久化存储多轮对话历史、用户个性化参数配置、Token 用量与成本统计。
- **Gemini Client (`gemini_client.py` & `mock_gemini.py`)**：集成 Google 官方 SDK，支持配置模型、Thinking Level (三档: LOW, MEDIUM, HIGH)、Google Search Grounding 实时检索与来源提取，以及全面的本地自动化测试 Mock 桩。

---

## 2. 法律与合规条款分析（重要）

依据 Google Gemini API 官方服务条款（Gemini API Additional Terms of Service）及区域开放名单：

1. **API 调用服务器在澳大利亚的合规性**：
   - **完全合规**。澳大利亚（Australia）位于 Google Gemini API 与 Google AI Studio 的官方支持地区（Supported Territories）名单中。在澳大利亚本地家庭宽带网络环境下部署并调用官方 API 完全符合服务条款。
2. **最终使用者地区的限制**：
   - Google 条款要求无论是直接开发者还是通过客户端使用 API 输出的最终用户，均应位于受支持的国家和地区内。
3. **中国大陆地区用户合规风险**：
   - 中国大陆目前**不在** Google Gemini API 官方开放地区名单中。
   - 依据服务条款，若境内熟人直接使用该服务，即便调用服务器架设在澳大利亚，亦存在区域合规瑕疵。
   - **合规指引**：在未取得相关授权前，系统提供 QQ 白名单严格隔离，建议仅在受支持地区（如澳大利亚本地测试账号）开放真实 Gemini API；对于境内用户，可先通过 Mock 模式或本地知识模式进行实验，切勿通过反向代理或伪造地区规避限制。
4. **非商业、低频私人实验性质**：
   - 个人开发者在 2-3 人熟人范围内的低频学习与实验，符合开发者个人评估及学习用途，不涉及转售、商业分发或代理盈利。
5. **免费层 (Unpaid Tier) 与付费层 (Paid Tier) 的数据隐私差异**：
   - **免费层 (Unpaid / Free of Charge)**：Google 会将提交的 Prompt 和模型生成内容用于产品改进与模型训练，并可能由人工审核员（Human Reviewers）对脱敏样本进行抽检标注。因此**切勿在对话中发送个人身份证件、密码、真实姓名等隐私或敏感机密信息**。
   - **付费层 (Pay-as-you-go)**：Google 明确承诺不会使用付费层用户提交的数据来训练模型，不进行人工抽检，数据隐私等级更高。

---

## 3. 安全与账号风险告知（NapCat / QQ 接入）

> [!WARNING]
> **非官方 QQ 协议风险提示**：
> NapCat / OneBot 11 采用模拟 NTQQ 客户端机制实现通信。腾讯安全风控系统对第三方机器人账号存在检测机制，使用非官方协议可能会导致 QQ 账号出现**临时风控、限制私聊/发消息、强制下线或账号冻结**。
>
> **安全防护策略**：
> 1. **严禁使用个人日常主账号**，必须使用专用的测试 QQ 小号进行登录。
> 2. **严格白名单控制**：系统默认只回复 `QQ_WHITELIST` 中配置的熟人账号，陌生人与任何群聊消息一律拦截忽略，避免任何批量触达与营销嫌疑。
> 3. **免公网暴露**：采用反向 WebSocket 架构（`reverse_ws`），本地应用仅监听 `127.0.0.1:8080`，NapCat 客户端主动连入，无任何端口映射或公网暴露风险。
> 4. **正式使用前请确认已知晓上述账号限制风险并授权**。

---

## 4. 快速部署与配置步骤

### 4.1 环境准备
本系统已在当前 Windows 11 主机上初始化专用虚拟环境：
- Python 版本：3.12.10
- 虚拟环境路径：`C:\Users\Zhengyang Yu\.gemini\antigravity\scratch\qq_gemini_assistant\.venv`

### 4.2 配置环境变量 (`.env`)
复制 `.env.example` 为 `.env`：
```powershell
Copy-Item .env.example .env
```
用文本编辑器打开 `.env` 并填写核心参数：
```ini
# Google Gemini 官方 API Key (从 Google AI Studio 获取)
GEMINI_API_KEY=AIzaSy...

# 默认模型 (支持 gemini-3.8-flash, gemini-3.5-flash-lite, gemini-4-argon)
GEMINI_MODEL=gemini-3.8-flash

# 默认思考强度 (三档: LOW, MEDIUM, HIGH)
DEFAULT_THINKING_LEVEL=MEDIUM

# 允许访问的 QQ 号白名单 (半角逗号分隔，只有白名单内的 QQ 号能触发回复)
QQ_WHITELIST=12345678,87654321

# 访问安全令牌 (与 NapCat 配置保持一致)
ONEBOT_ACCESS_TOKEN=my_secure_token_123
```

### 4.3 常用启动与图形化工具
项目根目录下提供了预配置好的图形化启动器与批处理脚本：
- **`QQGeminiAssistant.exe`**：**【推荐】Windows 原生 EXE 启动器**。直接双击即可启动图形化参数配置与控制中心。无需黑色 CMD 窗口，毫秒级静默响应，安全无误报。
- **`launcher.bat` / `launcher.vbs`**：批处理与 VBS 脚本启动入口（与 EXE 启动器功能一致）。
- **`run.bat`**：命令行方式直接启动智能助手后端服务（启动 OneBot 11 反向 WebSocket 服务端，等待 NapCat 连接）。
- **`console.bat`**：直接在 Windows 终端中进行交互模拟测试（无需启动 NapCat 或登录 QQ，可直接测试完整对话上下文、指令解析与模型响应）。
- **`test.bat`**：运行完整的自动化测试套件（Pytest）。
- **`stop.bat`**：一键安全停止所有正在后台运行的助手实例。
- **`build_exe.bat`**：重新编译生成 Windows EXE 启动器的构建脚本。

---

## 5. 指令与功能说明

### 5.1 完整可用指令列表（支持 `//` 或 `/` 前缀）
| 指令 | 说明 |
| :--- | :--- |
| `//help` 或 `/help` | 查看完整帮助说明及当前用户的专属个性化配置 |
| `//reset` 或 `/reset` | 清空当前用户的历史对话记忆 |
| `//status` 或 `/status` | 查看服务运行时长、并发限制、数据库状态与请求统计 |
| `//模型 [名称]` 或 `/model [名称]` | 查看或切换**当前用户的专属模型**（每个 QQ 用户独立生效，互不干扰） |
| `//搜索 开\|关` 或 `/search on\|off` | 为当前用户开启或关闭 Google Search 实时检索 |
| `//成本` | 实时查看个人与全局的 Token 消耗、搜索调用数与估算花费（美元/人民币） |
| `//低思考` / `//中思考` / `//高思考` | 调整当前用户的专属思考深度（默认低思考，保持快速流畅） |
| `//4轮对话记录` / `//8条消息` | 动态调整当前用户携带的历史消息深度（默认4条，即2轮） |
| `//帮我把模型换成3.5并开启低思考` | 模糊自然语言指令，由轻量模型智能解析并动态生效 |

### 5.2 回复排版与参考链接规则
- **纯文本无 Markdown**：系统提示词与回复排版层彻底禁用了 Markdown 语法（包括 `**加粗**`、`# 标题`、`* 列表`、````代码块```` 等），专为手机 QQ 聊天阅读优化。
- **按需提供链接**：为了保证聊天的清爽，机器人**绝不主动附加任何参考链接或网址**；仅当用户在提问中明确要求（如“给我链接”、“附带出处”、“参考网址是什么”等）时，才会提取并附带 Google Search Grounding 的来源链接。
- **免责声明**：涉及医疗、法律、金融风险话题时，自动在回复末尾附带免责声明提示。

---

## 6. NapCat (OneBot 11) 对接配置指引

如果决定在专用测试 QQ 小号上启用 QQ 实机连接：

1. 下载并安装 [NapCat Desktop](https://github.com/NapNeko/NapCat-Desktop) 或 Windows Shell 绿色版。
2. 打开 NapCat 网页控制面板（默认本机地址通常为 `http://127.0.0.1:6099/webui`）。
3. 进入 **网络配置** -> **添加通信配置** -> 选择 **反向 WebSocket (Reverse WebSocket)**：
   - **URL**: `ws://127.0.0.1:8080/onebot/v11/ws`
   - **Token**: 与 `.env` 中的 `ONEBOT_ACCESS_TOKEN` 一致（若留空则不填）。
   - **重连间隔**: `5000` 毫秒。
   - **心跳间隔**: `30000` 毫秒。
4. 使用手机 QQ 扫码登录**专用测试 QQ 账号**。
5. 启动本程序（双击 `run.bat`），NapCat 控制台将显示已成功连接至本地反向 WebSocket 服务端。
6. 使用白名单内的 QQ 好友账号向测试 QQ 发送私聊消息即可体验。

---

## 7. Windows 自动重启与服务保活配置

如需在开机时静默后台自启，可通过 Windows 任务计划程序 (Task Scheduler)：
1. 按 `Win + R` 输入 `taskschd.msc` 回车。
2. 点击右侧 **“创建任务”**：
   - 名称填写：`QQ_Gemini_Assistant`
   - 安全选项选择：“只在用户登录时运行”或“不管用户是否登录都要运行”。
3. 触发器标签：新建 -> 开始任务选择 **“登录时”** 或 **“启动时”**。
4. 操作标签：新建 -> 操作选择 **“启动程序”**：
   - 程序或脚本：`C:\Users\Zhengyang Yu\.gemini\antigravity\scratch\qq_gemini_assistant\run.bat`
   - 起始于：`C:\Users\Zhengyang Yu\.gemini\antigravity\scratch\qq_gemini_assistant`
5. 条件与设置标签：勾选“如果任务失败，按以下频率重新启动：每 1 分钟”，尝试重新启动最多 3 次。


## 启动器 v2 与模糊指令修复（2026-10-25）

双击项目根目录的 `QQGeminiAssistant.exe`，即可打开新版控制中心；也可使用 `launcher.bat`。
启动器仍依赖项目的 `.venv`，请保留整个项目目录，不要单独移动 EXE。

- 侧栏包含运行总览、参数配置、运行日志、用量统计、诊断工具。
- 先检查 API Key、QQ 白名单及 OneBot 地址，再启动服务；已运行时保存配置，下次启动生效。
- 支持完整的反向监听地址、端口、路径及正向 WebSocket URL。
- 参数校验失败不会写入配置或启动服务；停止服务不会阻塞窗口。
- 服务状态表示当前启动器管理的进程状态，QQ 是否连接请查看 OneBot 日志。
- 日志展示最多保留约 5000 行，隐藏当前配置中的 API Key 与 Access Token。

模糊指令失败的根因是 SQLite 用户配置允许 NULL：只修改模型后，历史条数字段仍可能为 NULL。
现在仅对 NULL 使用默认值，并保留合法的 0 和 False。“//换个笨一点的模型”会在本地识别，
从旗舰降至主力、主力降至轻量，轻量保持不变；其他复杂表达仍由 AI 提取配置。
该修复无需清空数据库或重置用户配置。原文件备份位于 `build/backup_20261025_122530/`。

回归验证：31 项测试通过，包括模拟 Gemini、数据库空值、启动器配置与启停、OneBot 本机 WebSocket。
未调用真实 Gemini API，未验证当前账号对各模型的访问权限或实际 QQ 收发。


## 图片接收修复（2026-10-25）

修复同一图片在结构化 `message` 与 CQ `raw_message` 中被重复提取的问题。
优先使用结构化图片；仅在没有有效结构化图片时回退 CQ 字符串。CQ 参数先拆分再还原
`&amp;`、`&#44;`、`&#91;`、`&#93;`，仅解码一层，保留 URL 签名的百分号编码。
同时对相同来源去重，保留不同图片的原始顺序。

验证：41 项测试全部通过，包含模拟 HTTP 下载及 OneBot → 图片暂存 → 文字追问 → 模拟 Gemini
的完整流程，以及图片与文字同发。未使用日志中的临时签名链接或请求真实 Gemini API。
重启机器人服务后生效；不需要清空数据库或修改 API Key。
图片代码原文件备份：`build/backup_image_20261025_124344/image_utils.py`。


## 指令帮助优化（2026-10-25）

`//help` 现按基础、模型、思考、搜索、记忆、输出、图片、用法八类提供完整说明，
包含可复制的命令、实际行为解释和当前账号配置。增加分类查询，例如 `//help 模型`、
`//help 图片`、`//help 当前`；`/help`、`//帮助` 等兼容入口同样支持分类。
帮助查询完全在本地生成，不修改用户配置、不消耗待问图片，也不调用 Gemini。
未知分类会提示可用分类，不会进入模糊指令解析。
重启机器人服务后生效。49 项回归测试全部通过。
本次代码备份：`build/backup_help_20261025_173829/chat_manager.py`。


## 思考强度误升档修复（2026-10-25）

修复本地正则回溯把“思考强度”的“强”误认为升档的问题。原先“思考强度设为低”、
“思考强度调成中”和“思考强度更低一点”都会错误地提升档位。
现在完整匹配思考术语，优先处理明确目标档位，正确区分相对升降；同时避免将
“降低思考”中的“低思考”和“提高思考”中的“高思考”误判为绝对档位。
支持中文及 LOW/MEDIUM/HIGH 明确档位，普通调节在本地完成。

示例：`//把思考强度设为低` → LOW；`//思考强度调成中` → MEDIUM；
`//思考强度更低一点` → HIGH 降至 MEDIUM，再发一次降至 LOW。
重启机器人服务后生效，可用 `//help 当前` 核对配置。
83 项回归测试通过，包含三档初始状态、逐级升降、边界保持、模型设置不受影响，
以及修改后的思考强度传入下一条模拟模型请求。未调用真实 Gemini API。
本次备份：`build/backup_thinking_20261025_184211/chat_manager.py`。
