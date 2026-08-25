# App 开发缺口清单（2026-08-25 检查）

> 定位：组长 2026-08-25 全面检查（84 个测试全绿 + 3 处必挂 bug 已修）之后，剩余待办。**每位成员开工第一件事：做完自己名下的缺口，打勾验收。**
> 验收命令默认在仓库根目录、conda 环境 `fl`（`D:\anoconda\envs\fl\python.exe`）执行。
> 铁律：冒烟输出写 `C:/tmp`；不动正式产物目录；控制台 GBK 安全字符。
> 已修复的不再列出：trainer 两处 bug（client 查找、one-hot 展开）、三个 bat 的 python 路径。

---

## A 的缺口（client_agent 全栈 + ε 链路）

### A-1（高）app/tests/ 三个测试文件缺失

**现状**：`app/tests/` 目录不存在。计划 Task 4/9/10 要求 `test_agent.py`（转发/登录/status）、`test_collector.py`（下载→解压→校验）、`test_trainer.py`（数据管线）。这次检查就是在缺测试的前提下靠手工冒烟才炸出 trainer 的两个 bug——必须补上防回归。

**做法**：按计划 Task 4/9/10 的测试代码写（已给完整代码），测试不依赖真实网络：
- test_collector：`monkeypatch urllib.request.urlopen`（局部 zip bytes）或直接用 `data/app_datasets/tetouan_0.zip` 本地文件转 `io.BytesIO`；断言 `dataset_id.txt`、rows、time_range
- test_trainer：**必须包含两条回归**——
  1. `start_training` 的 client 配置查找（`clients` 是 **dict**，用 `.get()`）能拿到 tetouan_city_0 的 sequences
  2. `build_train_cache` 在 CSV 只含 `category_id` 时能正确展开 one-hot（输入 100 行小 csv，`n_train > 0` 不抛 KeyError）

**合格标准**：
```
D:\anoconda\envs\fl\python.exe -m pytest app/tests/ -v
```
全部 PASS；无测试跳过真实网络的痕迹（无 urlopen 真请求）。

### A-2（高）/local/train-status 的 round/loss 永远为空

**现状**：`app/trainer.py` 的 `_state` 有 `round/loss` 字段，但全代码没有赋值点——`get_train_status()` 永远返回 `None`。前端客户端页"我的任务"的当前轮次/自身 loss 无法显示，录演示视频时是空白。

**做法**：给 `FedClient` 挂每轮回调。最简方案：在 `trainer.py` 里子类化/包装 `FedClient.fit`，每次 fit 后更新 `_state["round"]`（从 fit 参数或累计计数）与 `_state["loss"]`（fit 返回的 metrics["loss"]），并连同 `eps`。参考 `fl_code/fed_core/client_core.py` 的 `FedClient.fit`/`CidEchoClient` 结构（metrics 已在 server_core 侧消费，这里只是旁路记录）。

**合格标准**：本机起一个 **1 轮** flwr 服务器冒烟（拉到 `C:/tmp`），agent `/local/start` 后轮询：
```
curl http://localhost:9001/local/train-status
```
训练期内返回 `{"running": true, "round": >=1, "loss": <非空>, ...}`；训练后 `running: false`。

### A-3（中）Task 6 的 fed_core 层 ε 上报单测缺失

**现状**：ε 落库链路（FedClient.fit 报 eps → AuditFedAvg.aggregate_fit 收集 → client_epsilons 落库）只有 API 层测试（test_admin_dashboard.py），计划要求 `test_fed_core_epsilon_report.py` 覆盖 fit→aggregate 层。该链路是"每客户端隐私预算曲线"的数据源，值得锁死。

**做法**：在 `test/` 下新建测试：仿照 `test/test_fed_core_server.py` 的两客户端模拟，result.metrics 带 `{"cid": ..., "eps": 1.25}`——断言 `AuditFedAvg.audit_rows[0]["client_epsilons"]` 等于 `{"<cid>": 1.25}`；不含 eps 的 result 不影响其他行。

**合格标准**：
```
D:\anoconda\envs\fl\python.exe -m pytest test/ -v
```
新增用例 PASS，原 40 个不回归。

### A-4（低）app/api.py 死代码

**现状**：tkinter 时代遗留，全仓库无任何 import（已 grep 验证）。

**做法**：`git rm app/api.py`。

**合格标准**：删除后 `pytest server/tests/ app/tests/` 仍全绿。

---

## B 的缺口（前端 + 手测记录）

### B-1（高）客户端页手测清单无记录

**现状**：Task 3/11 是手动验收，但前端从未过一遍完整流程，无记录。

**做法**：本机跑 `app/run_client.bat`（agent）→ 浏览器 localhost:9001 按下面清单走一遍，F12 console 无红色报错，结果写在 `C:/tmp/b_手测记录.txt`：

1. 未登录时页面提示登录；登录页注册→登录成功进广场
2. agent_config.json 未配 server_url 或 server 未启动时：页顶"客户端代理未连接"提示出现（页面本身能打开，ECharts 不报错）
3. 广场只显示招募中任务；"发起任务"弹窗能创建；"凭密钥加入"弹窗用错密钥有报错提示
4. 我的任务：显示参与/发起的任务列表；"采集数据"按钮弹数据源列表（来自 /api/datasets 或本地兜底），采集成功显示"N 条记录，时间范围，缺失率"
5. 采集后"开始训练"按钮可用（此时若无服务器，应提示连接状态而非 JS 崩溃）
6. 切到管理端页面（server:8000）走一遍登录→广场，确认一套代码两处渲染均正常

**合格标准**：清单 1-6 全过 + `C:/tmp/b_手测记录.txt` 上注明每项"通过/失败原因"；发现的问题转缺口（截图或描述）。

---

## C 的缺口（server API + 管理端大屏）

### C-1（高）大屏 6 图手测无记录

**现状**：Task 8 实现了 6 图（ch-participants/heatmap/eps/loss/clip + RC 区），但没有任何实际数据跑过的大屏验证记录。

**做法**：server 起（`run_server.bat`）→ 浏览器 :8000 → admin/admin123 登录 → 选一个**有 audit 数据**的任务进详情大屏：

1. 数字卡片：任务名/状态/轮次 (a)/参与人数 (b)
2. 每轮参与人数曲线非空；热力图绿=join/红=dropped/灰=未注册，横轴轮次纵轴客户端
3. 每客户端累计 ε 曲线非空（对应 audit 的 client_epsilons，若某任务没有则标注原因不判过）
4. 全局 loss 曲线、自适应裁剪阈值 C 曲线（clip_norm，若任务非 adaptive-clip 则留空=预期）
5. RC 区：上传过结果的客户端有 WAPE 双柱 + 对比图 PNG；没结果时显示"暂无"
6. 轮询 2.5s：训练中的任务能看到轮次更新（对 completed 任务至少不报错）
7. 普通用户账号登录后不能进大屏（无 admin 权限提示）

**合格标准**：1-7 全过 + `C:/tmp/c_大屏手测.md`（每项通过/失败 + 截图路径）。服务端 44 个测试已绿，此为功能层验收。

---

## 组长组（与同学无代码交叠）

### L-1（高，组长）GitHub push ✅ 已完成 2026-08-25

push 完成，4 个 raw URL 已验证返回 200（各 zip 字节数与本地一致，Windows 侧验证）。

### L-2（高，组长）docs/App部署手册.md ✅ 已完成 2026-08-25

已建 `docs/App部署手册.md`：方案 A 公网（云服务器 2C4G + Docker Compose、安全组放行 8000/8089）+ 方案 B 局域网（Windows run_server.bat、防火墙规则）、客户端五步上手、5 分钟演示剧本、排查表、端口/账号速查、安全声明。

### L-3（低，组长或任意成员）CRLF 行尾符假 diff 清理 ✅ 已完成 2026-08-25

`.gitattributes` 已建（`*.py/js/md/json/yaml/csv/svg` 等 eol=lf，`*.bat` eol=crlf，`zip/pt` binary），`git add --renormalize .` 已执行，commit `9a01058`。**后续 Windows 编辑器保存后 git status 不再出现假 diff；新文件也按此约定写入**（注意：*.bat 保持 CRLF 正确，勿用其他工具改成 LF）。

### L-4（组长）分发 + 提交清单

- 把 `docs/superpowers/specs/2026-08-21-*.md` + `docs/superpowers/plans/2026-08-21-*.md` 发给 A/B/C（gitignore 里，git 拉不到）
- 各人验收材料（C:/tmp 手测记录）收齐后写提交清单（Task 16）

---

## 全体联调（Task 15，高）

### T-1 真实网络端到端

**现状**：代码级链路全通（采集→管线→server 测试→agent 冒烟），但从未在真实网络（server+2 客户端远程）跑通全程。

**流程**：部署手册（L-2）发布后，三台机器按流程：server 起 → admin 发起任务 → 客户端 1/2 各 run_client.bat → 登录 → **采集（真实 GitHub URL）** → 凭密钥加入 → start → 训练 ≥3 轮（让 1 台中途掉线演示 audit dropped 记录）→ 完成 → 客户端下载模型 → 本地 RC → 上传 → 管理端大屏查看 WAPE + 对比图。

**合格标准**：
1. 大屏全程可见轮次/参与/ε/loss 更新
2. 客户端页显示当前轮次/loss（依赖 A-2）
3. 掉线客户端在 audit 的 dropped 里如实记录，训练不中断（accept_failures）
4. RC 上传后大屏 RC 区出现数据
5. 全程无未处理异常（server 日志无 Traceback）

**打回条件**：任意一步失败即未达标，失败信息（截图+日志尾部）补到缺口清单。

---

## 验收优先级说明

全部按"高"排期（08-25~08-31），因为：

- A-1/A-2/T-1 决定 09-01~09-05 联调（阶段 4）能否按时
- L-1 是 A/C 采集功能的前置（push 后 URL 才活）
- B-1/C-1 手测是 09-06~09-14 录视频的前置（没验收过的功能不能拍进视频）
