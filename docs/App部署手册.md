# App 部署手册（Web 版）

> 版本：v1.0（2026-08-25）
> 适用范围：把"联邦学习 + 差分隐私 + 个性化修正"App 部署到真实网络，供双机/云端演示。
> 相关：`docs/App开发交接说明.md`（架构与分工）、`data/app_datasets/README.md`（数据源 URL，已生效）、`docs/App缺口清单.md`（联调项 T-1）。

## 0. 整体图景

```
server 机器（公网云 / 局域网机）          客户端机器（每台都装 agent）
┌─────────────────────────────┐        ┌──────────────────────────────┐
│ REST :8000 (FastAPI)        │        │ run_client.bat (agent :9001) │
│ flwr gRPC :8089             │◄──────►│ 浏览器 localhost:9001 页面     │
│ 前端静态页（管理端大屏）      │        │ 数据采集→训练→RC               │
│ SQLite 数据库                │        │（依赖 :8000 与 :8089 连通）   │
└─────────────────────────────┘        └──────────────────────────────┘
```

关键端口：**8000（REST/网页）**、**8089（flwr 训练）**——两台都必须在防火墙上放行。

## 1. 方案 A：公网部署（推荐演示用）

| 项 | 建议 |
|---|---|
| 服务器 | 云厂商轻量应用服务器 2C4G（Ubuntu 22.04/24.04） |
| 部署方式 | Docker Compose（一条命令起全套） |
| 客户端 | 各自本机 Windows + conda `fl` 环境（见 §2.3） |
| server_url | `http://<服务器公网IP>:8000` |

### 1.1 服务器初始化（Ubuntu 云服务器，一次性）

```bash
# 1. 安装 Docker（已有则跳过）
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# 2. 拉代码（需 GitHub 访问；build context = 仓库根）
git clone https://github.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System.git
cd Personalized-Federated-Learning-Power-Forecasting-System

# 3. 构建并启动（首次构建约 5-10 分钟，torch CPU 版，无 GPU 依赖）
docker compose up --build -d
docker compose logs -f fl-server     # 看到 uvicorn 启动日志即成功
```

### 1.2 云安全组放行（每个云厂商界面不同，步骤同义）

1. 控制台 → 实例 → 安全组/防火墙 → 添加入站规则
2. 协议 TCP，端口 **8000**（来源 0.0.0.0/0，演示期全放行；演示完改小）
3. 同上添加 TCP **8089**
4. 服务器内系统防火墙（若有）：`sudo ufw allow 8000/tcp && sudo ufw allow 8089/tcp`

### 1.3 验证

```bash
curl http://<公网IP>:8000/api/health      # → {"status":"ok"}
# 浏览器打开 http://<公网IP>:8000 → 登录页（管理端）
```

> 公网方案注意：8089 为明文 gRPC，演示级不做 TLS；公网演示完请立刻收紧安全组。

## 2. 方案 B：局域网部署（双机/教室演示）

| 项 | 建议 |
|---|---|
| server 机器 | 组长机（Windows 已装 conda `fl`）或机房任意一台 |
| 客户端 | 同局域网的其他 Windows 电脑 |
| server_url | `http://<server机局域网IP>:8000` |

### 2.1 server 机启动（Windows，双击）

1. 确认 `D:\anoconda\envs\fl\python.exe` 存在；不存在则先装环境（§2.3）
2. 确保本机 8000/8089 未被占用
3. **双击 `run_server.bat`** → 窗口显示 `Uvicorn running on http://0.0.0.0:8000`，保持窗口常开
4. 查到本机局域网 IP：`ipconfig`（IPv4 地址，如 `192.168.1.20`）

### 2.2 Windows 防火墙放行

弹窗时允许，或手动：

```
控制面板 → Windows Defender 防火墙 → 高级设置 → 入站规则 → 新建规则
  → 端口 → TCP 8000 与 8089 → 允许连接 → 命名"FL-App"
```

> 放行后server机仍连不上时：用 `netstat -ano | findstr 8089` 确认服务真的在听；关掉"专用网络"防火墙开关试一次（排除规则干扰后记得开回来）。

### 2.3 客户端准备（每台参与机）

```bash
# 项目根目录（git 已 clone / 解压）
D:\anoconda\envs\fl\python.exe -m pip install -r server/requirements.txt
D:\anoconda\envs\fl\python.exe -m pip install "flwr" "torch" --index-url https://download.pytorch.org/whl/cpu
```
（有条件直接 `conda env create -f environment.yml && conda activate fl` 更省事。）

**配置 `app/agent_config.json`**：只改 `server_url`，其余保持（账号在页面登录时输入，会把 username/password/client_id 写进本文件）：

```json
{
  "server_url": "http://192.168.1.20:8000",
  "username": "",
  "password": "",
  "client_id": "tetouan_city_0",
  "local_port": 9001
}
```

`client_id` 可选，但**推荐预填**，防止采集错数据集（采集时会校验"数据源属于哪个客户端"）：

| client_id | 采集哪个数据源 |
|---|---|
| `steel_ind_0` | steel_ind_0（钢铁厂整份） |
| `tetouan_city_0` | tetouan_0（Zone1 工业区） |
| `tetouan_city_1` | tetouan_1（Zone2 混合区） |
| `tetouan_city_2` | tetouan_2（Zone3 居民区） |

## 3. 客户端五步上手（每台客户端机）

1. **双击 `app/run_client.bat`** → 自动开浏览器 `http://localhost:9001`（控制台窗口保持常开）
2. **登录/注册**：页面上注册一个普通账号（用户名随意）→ 登录
3. **采集数据**：我的任务（或广场入口）→"采集数据"→ 选中数据源 → 完成后显示"N 条记录 / 时间范围 / 缺失率"（数据落 `app/data/`，需 GitHub 可达，只下载一次）
4. **加入任务**：广场 → 有招募中的任务 → 点"凭密钥加入" → 填发起者给的**密钥**（此步记录自己的 client_id）
5. **等发起者点"开始训练"** → 自动开训；期间可看本地状态；训练完点"训练残差修正器"（RC），完成后自动上传指标与对比图

## 4. 演示剧本（5 分钟版，双机）

| 时间 | 操作 | 画面 |
|---|---|---|
| 0:00-0:30 | server 机双击 run_server.bat；浏览器 :8000 登录 **admin/admin123** | 管理端登录页 |
| 0:30-1:00 | admin（或任一账号）"发起任务"：轮次填 20-30、时长 60s/轮 → 生成**密钥**复制 | 任务创建成功弹窗 |
| 1:00-1:30 | 客户端机：登录 → 采集数据（点选 tetouan_0 → 完成提示）→ 凭密钥加入 | 采集成功统计 |
| 1:30-2:30 | 发起者点"开始训练"；**大屏切到任务详情** | 6 图逐轮刷新（轮次/参与/ε/loss/裁剪阈值） |
| 2:30-3:00 | 中途关掉一个客户端窗口（演示掉线）| 参与热力图变红、dropped 如实记录，训练不中断 |
| 3:00-4:00 | 训练结束 → 客户端点"训练残差修正器" | 本地 RC 训练 → 自动上传 |
| 4:00-4:30 | 大屏刷新 RC 区 | WAPE 双柱（全局 vs 全局+RC）+ 对比图 |
| 4:30-5:00 | 收尾：管理端大屏全景 + 客户端"我的任务"页 | 隐私预算曲线、loss 曲线 |

排练建议：正式演示前至少完整跑 1 遍（不中断版本），确认数据源 URL 可下载、端口全通。

## 5. 常见问题排查表

| 现象 | 排查 |
|---|---|
| 双击 run_server.bat 窗口闪退 | conda 路径不对（应为 `D:\anoconda\envs\fl\python.exe`）或 8000 被占；改 bat 里路径、`netstat -ano \| findstr 8000` |
| 浏览器打不开 server:8000 | server 未起来（看窗口日志）；URL 写错（http 别漏）；防火墙未放行 |
| 客户端页面能开但提示"无法连接远程服务器" | `agent_config.json` 的 server_url 不对；或 server 机防火墙 8000 未放行 |
| admin 登录失败 | seed admin 默认 `admin/admin123`；`ADMIN_USERNAME/ADMIN_PASSWORD` 环境变量覆盖时以 env 为准 |
| "未知数据集"或采集 404 | 数据源 raw URL 未生效——确认已 push GitHub（`curl -I <url>` 返回 200；**客户端机若连不上 raw.githubusercontent.com**，演示前至少提前采集一次） |
| 训练掉线/客户端中途退出 | 属预期演示项：audit 会如实记录 dropped；连续掉线会触发 round_timeout，训练照常收尾 |
| 发起训练报"已有一个任务在训练" | flwr 8089 一次只支持一个训练任务；等前一个任务完成或重启 server |
| 客户端加入时说密钥错误 | 密钥只在创建时显示一次（server 存哈希）；重新创建任务取新密钥 |
| 中文乱码 | 控制台 GBK：窗口标题/提示均要求 GBK 安全字符，不要用 σᵢ 等下标字符 |

## 6. 端口与账号速查

| 名字 | 值 |
|---|---|
| REST/网页 | `:8000` |
| flwr gRPC | `:8089`（同时只能跑一个训练任务） |
| 客户端代理 | 本机 `:9001` |
| 数据库文件 | `data/fl_server.db`（server 机；删掉 = 重置全部数据，重新 seed admin） |
| admin 账号 | `admin/admin123`（演示级，环境变量可改） |
| 客户端数据目录 | `app/data/`（采集落盘；训练后含 `rc_work/` `rc_out/`） |

## 7. 安全声明（写进汇报材料）

- 客户端原始用电数据永不离开本机（仅上传 DP 加噪后的模型参数与 RC 指标/对比图）
- 模型参数仅存 server 内存/DB 审计元数据；任务密钥只存哈希
- 本轮实现为演示级：明文 gRPC + 简单口令，实际生产需 TLS 与正规认证（已属加分项范畴）
