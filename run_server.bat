@echo off
title FL-Server-8000
rem Start FL platform server on port 8000
rem Double-click to run. Keep this window open while testing.
cd /d %~dp0

rem flwr worker 的运行目录（日志 + fl_model_<id>.pkl）。
rem fl_runner 默认写 "/app/data"（Docker 容器内路径），裸机 Windows 上会解析成
rem D:\app\data 且不存在，导致点"开始训练"时写 worker 日志直接 FileNotFoundError。
rem 这里显式指向仓库内的 data 目录（Docker 部署不需要此脚本，走 /app/data 卷）。
set MODEL_DIR=%~dp0data
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

rem Python 路径兜底：本机装了 ml 环境就用它，否则退回 PATH 里的 python
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m uvicorn server.main:app --host 0.0.0.0 --port 8000
pause
