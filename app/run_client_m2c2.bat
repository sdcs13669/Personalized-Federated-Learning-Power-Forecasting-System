@echo off
title FL-Agent-机器2-客户端2-9002 (tetouan_city_0)
rem 机器2 · 客户端 2/3：tetouan_city_0，端口 9002，数据目录 app\data_m2c2（已预采集）
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_m2c2.json
set FL_DATA_DIR=%~dp0data_m2c2
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9002
"%PY%" app\agent.py
pause
