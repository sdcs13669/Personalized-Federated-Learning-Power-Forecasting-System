@echo off
title FL-Agent-数据组2-客户端2-9002 (tetouan_city_0)
rem 数据组2 · 客户端 2/3：tetouan_city_0，端口 9002，数据目录 app\data_m2c2
rem 注意：脚本名里的 m1/m2 是【数据组】编号，与"哪台机器"无关。
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_m2c2.json
set FL_DATA_DIR=%~dp0data_m2c2
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9002
"%PY%" app\agent.py
pause
