@echo off
title FL-Agent-1B-9002 (tetouan_city_0)
rem 演示机器1 · 客户端 B：tetouan_city_0，端口 9002，数据目录 app\data_1b
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_1b.json
set FL_DATA_DIR=%~dp0data_1b
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9002
"%PY%" app\agent.py
pause
