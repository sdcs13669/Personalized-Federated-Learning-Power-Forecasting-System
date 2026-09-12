@echo off
title FL-Agent-机器2-客户端3-9003 (tetouan_city_1)
rem 机器2 · 客户端 3/3：tetouan_city_1，端口 9003，数据目录 app\data_m2c3（已预采集）
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_m2c3.json
set FL_DATA_DIR=%~dp0data_m2c3
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9003
"%PY%" app\agent.py
pause
