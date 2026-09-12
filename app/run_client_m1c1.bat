@echo off
title FL-Agent-机器1-客户端1-9001 (steel_ind_0)
rem 机器1 · 客户端 1/3：steel_ind_0，端口 9001，数据目录 app\data_m1c1
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_m1c1.json
set FL_DATA_DIR=%~dp0data_m1c1
rem Python 路径兜底：本机装了 ml 环境就用它，否则退回 PATH 里的 python
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9001
"%PY%" app\agent.py
pause
