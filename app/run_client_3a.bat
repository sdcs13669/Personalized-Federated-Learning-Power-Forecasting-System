@echo off
title FL-Agent-3A-9001 (lcl_res_0)
rem 演示机器3 · 客户端 A：lcl_res_0，端口 9001，数据目录 app\data_3a
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_3a.json
set FL_DATA_DIR=%~dp0data_3a
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9001
"%PY%" app\agent.py
pause
