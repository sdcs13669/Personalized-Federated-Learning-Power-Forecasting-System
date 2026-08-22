@echo off
rem 启动 client_agent（本地 Web 代理，端口 9001）
rem Python 路径按机器环境修改：组长机器 D:\anoconda\envs\fl\python.exe
cd /d %~dp0..
D:\Miniconda\envs\power\python.exe app\agent.py
start http://localhost:9001
