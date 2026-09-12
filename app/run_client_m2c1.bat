@echo off
title FL-Agent-数据组2-客户端1-9001 (steel_ind_0)
rem 数据组2 · 客户端 1/3：steel_ind_0，端口 9001，数据目录 app\data_m2c1
rem 注意：脚本名里的 m1/m2 是【数据组】编号，与"哪台机器"无关 ——
rem 交换测试时本组可以换到另一台机器上跑（同一时刻每台机器只能开一组）。
cd /d %~dp0..
set FL_AGENT_CONFIG=%~dp0agent_config_m2c1.json
set FL_DATA_DIR=%~dp0data_m2c1
set PY=D:\anaconda3\envs\ml\python.exe
if not exist "%PY%" set PY=python
start "" http://localhost:9001
"%PY%" app\agent.py
pause
