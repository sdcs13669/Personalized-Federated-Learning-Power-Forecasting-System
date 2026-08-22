# App 演示数据集

为 App 端"模拟数据采集"准备的 4 个数据源。每个 zip 内有一个同名 CSV（列结构与 `data/processed/` 一致；tetouan 各 zone 文件只保留本 zone 负荷列）。

| 数据集 id | 客户端 | 内容 | 行数 | zip 大小 |
|-----------|--------|------|------|----------|
| steel_ind_0 | steel_ind_0 | 钢铁厂用电，整份（365 天） | 17520 | 386K |
| tetouan_0 | tetouan_city_0 | 城市用电 Zone1（工业区，364 天） | 17472 | 540K |
| tetouan_1 | tetouan_city_1 | 城市用电 Zone2（混合区，364 天） | 17472 | 542K |
| tetouan_2 | tetouan_city_2 | 城市用电 Zone3（居民区，364 天） | 17472 | 541K |

## 下载 URL（push 到 GitHub 后生效）

仓库：`https://github.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System`，分支 `main`，目录 `data/app_datasets/`：

```
https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/steel_ind_0.zip
https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_0.zip
https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_1.zip
https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_2.zip
```

Task 7 的 `server/routers/datasets.py` 中 DATASETS 清单直接复制上述 URL。

## 重新生成

```bash
D:\anoconda\envs\fl\python.exe data/app_datasets/prepare_app_datasets.py
```

## 注意事项

- 上传动作：`git push` 到 `main` 后 URL 才可访问；push 前可用 `curl -I <url>` 自测
- 演示现场可提前采集（下载一次即落到本地 `app/data/`），断外网不影响已采集数据
- 时间范围：steel 2018-01-01~2018-12-31；tetouan 2017-01-01~2017-12-30
