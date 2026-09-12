@echo off
title FL-Agent-3B-9002 (lcl_res_1)
rem 演示机器3 · 客户端 B：lcl_res_1，端口 9002，数据目录 app\data_3b
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_3b.json
set FL_DATA_DIR=%~dp0data_3b
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9002
"%PY%" app\agent.py
pause
