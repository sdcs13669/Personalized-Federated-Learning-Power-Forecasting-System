# App 开发交接说明（Web 版）

> 读者：App 线 3 名同学（A/B/C）
> 版本：v2.0（2026-08-25，全面 Web 化修订；取代 2026-08-16 tkinter 壳子版）
> 关联文档：设计 `docs/superpowers/specs/2026-08-21-fl-app-web-redesign-design.md`、计划 `docs/superpowers/plans/2026-08-21-fl-app-web-redesign.md`（16 Task）、缺口清单 `docs/App缺口清单.md`（当前待办，先看这个）——三者均为主文件，后者在 docs/superpowers/ 下的不在 git 里，由组长单独分发（见 §8）

## 1. 这个 App 是干什么的

一句话：**让多台电脑一起训练一个用电量预测模型，但每台电脑的数据始终留在自己机器上，不出门。**

- 一台"管理机"（服务器）承载任务广场：任何人可发起任务，其他人凭密钥加入
- 每台客户端用**自己本机的电力数据**在本地训练，只上传模型参数（不是数据！），服务器聚合后发回，如此往复多轮
- 隐私保护：客户端上传前做差分隐私（DP）加噪（本地裁剪 + 高斯噪声），服务器拿不到任何客户端的原始数据
- 管理端大屏实时展示：轮次、参与人数、参与热力图、每客户端隐私预算 ε、全局 loss、自适应裁剪阈值 C
- 训练结束后：每台客户端下载全局模型 → 本地训练"残差修正器"（个性化，纯本地）→ 上传 WAPE 指标 + 对比图 → 大屏展示"全局模型 vs 全局+RC"精度对比

### 术语速查（写代码/问 AI 时会遇到）

| 术语 | 大白话解释 |
| --- | --- |
| 联邦学习 | 多台机器各用自己的数据训练，只交换模型参数、不交换数据，最后合成一个全局模型 |
| 聚合（FedAvg） | 服务器把各客户端上传的参数按数据量加权平均，得到新的全局模型 |
| 差分隐私（DP） | 给上传的参数加随机噪声，让服务器无法反推任何一个人的数据；ε/δ 是"花了多少隐私预算"的参数，C 是裁剪范数（限制单个样本的影响） |
| 残差修正器（二阶段 RC） | 每台客户端本地训练的小模型，专门修正全局模型在自己数据上的预测误差，纯本地、不上传 |
| flwr / gRPC | flwr（flower）是现成的联邦学习框架，底层用 gRPC 通信协议——App 只调用它，不用自己实现网络协议 |
| client_agent | 每台客户机上的本地代理进程（`app/agent.py`），托管网页 + 转发 API + 采集数据 + 跑训练 |

## 2. 架构（2026-08-21 转向 Web 后）

```
┌──────────────────────────────────────────────────────┐
│ Server（云服务器 / 机房机器，Docker 或裸机）            │
│  FastAPI REST :8000 + 托管前端静态页（管理端大屏）      │
│  flwr server 后台线程 + SQLite（用户/任务/审计/RC）     │
└───┬────────────────────────────┬─────────────────────┘
    │ REST :8000                  │ gRPC :8089
┌───┴───────────┐        ┌───────┴────────────────────┐
│ 浏览器（管理端）│        │ 浏览器（客户端）              │
│ 登录 admin     │        │ http://localhost:9001      │
│ 任务广场+大屏   │        │ 广场/我的任务/采集/训练控制   │
└────────────────┘        └───────┬────────────────────┘
                                  │ HTTP 同机
                     ┌────────────┴────────────────────┐
                     │ client_agent.py（双击 run_client.bat）│
                     │ · 数据采集（下载+解压+校验）        │
                     │ · flwr 客户端训练+每轮上报          │
                     │ · 二阶段 RC + 上传指标/图           │
                     │ · 转发层（页面请求→注入 JWT→server） │
                     └─────────────────────────────────┘
```

关键点：

- **一套前端代码两处部署**：`web/` 目录，server 挂载 = 管理端（admin 大屏）；agent 挂载 = 客户端。页面通过 `/local/status` 自动识别自己处于哪种模式
- **管理端角色 = admin**：独立账号（seed 默认 admin/admin123，示范级），可看**所有状态**任务并进大屏；普通用户广场只显示招募中的任务
- **JWT 不经过浏览器**：客户端页面的所有 `/api/*` 请求发给 localhost:9001 的 agent，agent 转发到远程 server 并注入 token——浏览器代码不感知 token

## 3. 老师 6 项要求的落点

| # | 老师反馈 | 落点 |
|---|---------|------|
| 1 | 模拟数据采集 | 页面点"采集数据"→ 选数据源（GitHub raw URL，4 个 zip 已就绪）→ agent 下载解压校验 → 本地 `app/data/` 即训练数据 |
| 2 | 管理员界面 | admin 登录 → 任务信息表 → 任务详情大屏：轮次/参与人数曲线/参与热力图/每客户端累计 ε/全局 loss/裁剪阈值 C（6 图）+ RC 结果区 |
| 3 | 客户端界面 | 登录 → 广场（发起任务/凭密钥加入）→ 我的任务（参与的任务 + 自身状态 + 采集/训练/RC 控制） |
| 4 | 二阶段 RC 可视化 | 训练完后端"训练残差修正器"→ 上传 WAPE 对比（全局 vs RC）+ 预测曲线对比图 PNG → 大屏三层展示 |
| 5 | 美化/卡顿 | 全面 Web + ECharts（本地 vendor 文件，断外网也能渲染） |
| 6 | 真实网络 FL | server 地址可配置（agent_config.json），部署手册有公网/局域网双方案 |

## 4. 分工与任务清单（最新）

所有开发归 A/B/C；组长只负责文档 + 把关 + GitHub 上传。**任务细节、完整代码都在实施计划里，对号入座：**

| 成员 | 负责 Task | 内容 |
| --- | --- | --- |
| A | 1/4/6/9/10/12 | 数据集脚本（✅）· client_agent 全栈（转发/采集/训练/RC）· ε 落库 |
| B | 2/3/11 | server 静态托管 · 前端骨架 · 客户端页（我的任务/采集/训练控制） |
| C | 5/7/8/13 | admin 角色 + my/tasks + datasets/RC 接口 · 管理端大屏 6 图 + RC 展示 |
| 组长 | 14/16 | 部署手册 · 交接说明/提交清单 · 联调把关 · GitHub 上传 |

各成员再开工前，**先看 `docs/App缺口清单.md`**——2026-08-25 检查发现的问题（含合格标准）都在里面，优先补齐。

## 5. 数据源（Task 1 已完成）

4 个 zip 已提交 git 并写入 raw URL（`data/app_datasets/README.md`），**需组长 push GitHub 后 URL 才可访问**：

| 数据集 id | 客户端（client_config 里） | 内容 |
| --- | --- | --- |
| steel_ind_0 | steel_ind_0 | 钢铁厂用电整份（365 天） |
| tetouan_0 | tetouan_city_0 | 城市 Zone1 工业区 |
| tetouan_1 | tetouan_city_1 | 城市 Zone2 混合区 |
| tetouan_2 | tetouan_city_2 | 城市 Zone3 居民区 |

采集后在 `app/data/` 落盘；client_id 与 client_config.yaml 的映射见 `server/routers/datasets.py` 的 DATASETS 清单。

## 6. 与实验线的分工边界

- 训练核心只用 `fl_code/fed_core`（client_core/server_core/params/accounting）——**唯一真源，不要重写训练逻辑**；DP 在客户端本地做，上传前加噪
- 二阶段 RC 复用 `fl_code/train_personalized.py`（已支持 `--data-dir` 指向 app 采集目录；实验线默认行为不变）
- 数据管线复用 `fl_code/data_utils.py`（preprocess/split_train_test/make_sliding_windows/PowerDataset）；注意它要求 `category_id` 展开为 cat_* 公共特征，`app/trainer.py` 已对齐
- server 数据已齐：audit_rounds 有 round/expected/joined/dropped/loss/client_losses/client_epsilons/clip_norm——大屏 6 图全部有原始数据

## 7. 怎么用 AI coding agent 干活

1. **喂三份材料（顺序）**：① 本说明 + 缺口清单（知道做什么）→ ② 实施计划里你自己的 Task（直接说"读计划 Task N，按步骤执行"）→ ③ 设计文档（有疑问时查"为什么"）。**先让组长把 docs/superpowers/ 下的文件发到你机器上**（§8）
2. **让 agent 严格按计划步骤执行**：先写测试 → 确认失败 → 写实现 → 确认通过 → 提交，别跳步。界面细节计划里有完整代码，照抄即可
3. **验收**：`pytest server/tests/`（C）+ `pytest app/tests/`（A）+ 手动按缺口清单的"手动验收"节点一遍
4. **铁律**（计划 Global Constraints 里也有）：
   - 冒烟/验证必须把输出指到 `C:/tmp` 临时目录，**严禁写正式产物目录**（fl_code/baseline_outputs、personalized_outputs、figures、app/data 默认路径）
   - 控制台打印只用 GBK 安全字符（Windows 控制台），禁用下标字符（σᵢ 等）
   - Python 用 `D:\anoconda\envs\fl\python.exe`（不是 `D:\anaconda3\envs\ml`——那是个坑，已修）
   - 前端不依赖 CDN（ECharts 用 web/vendor/echarts.min.js，演示现场断外网也能渲染）
   - 提交前 `git status` 检查，别把运行产物提交进去

## 8. 文档地图

| 文档 | 内容 | 什么时候看 |
| --- | --- | --- |
| 本说明 | 愿景 + 架构 + 分工 | 先看这个 |
| `docs/App缺口清单.md` | **当前待办**（缺口 + 合格标准） | 开工前必看 |
| 设计文档 `docs/superpowers/specs/2026-08-21-fl-app-web-redesign-design.md` | 架构决策、server 改动、agent 端点、前端结构 | 有疑问时查 |
| 实施计划 `docs/superpowers/plans/2026-08-21-fl-app-web-redesign.md` | 16 个 Task，逐步步骤含完整代码 | 干活时用 |
| `data/app_datasets/README.md` | 4 个数据源 raw URL | 采集接口要填 URL 时 |
| CLAUDE.md | 仓库总说明、命令、环境 | 环境配置时看（注意：其 `tests/` 目录实际为 `test/`） |

> 注意：docs/superpowers/ 在 .gitignore 里（git 拉不到），由组长单独发文件；设计/计划/本说明/缺口清单是"读文件给 agent"的输入，勿直接复制粘贴进聊天。

## 9. 时间线

| 日期 | 里程碑 |
| --- | --- |
| 08-22 | Task 1 数据源就绪（4 zip 已提交） |
| 08-23~08-24 | 阶段 1-3 功能提交（B 前端、C 大屏、A agent——已合并） |
| 08-25 | 组长检查 + 修复 3 个必挂 bug + 出缺口清单 |
| 08-26~08-31 | **补缺口**（A 的 app 测试/轮次回调、B/C 手测记录）+ 组长 push GitHub |
| 09-01~09-05 | 阶段 4：二阶段 RC 联调 + 部署手册 + 真实网络端到端（Task 15） |
| 09-06~09-14 | 阶段 5：彩排、录演示视频、提交清单（Task 16） |
| 09-15 | 作品提交 |
