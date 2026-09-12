@echo off
title FL-Agent-2A-9001 (tetouan_city_1)
rem 演示机器2 · 客户端 A：tetouan_city_1，端口 9001，数据目录 app\data_2a
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_2a.json
set FL_DATA_DIR=%~dp0data_2a
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9001
"%PY%" app\agent.py
pause
