@echo off
title FL-Agent-2B-9002 (tetouan_city_2)
rem 演示机器2 · 客户端 B：tetouan_city_2，端口 9002，数据目录 app\data_2b
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_2b.json
set FL_DATA_DIR=%~dp0data_2b
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9002
"%PY%" app\agent.py
pause
