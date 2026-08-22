@echo off
title FL-Agent-9001
rem Start local client agent (web proxy on port 9001)
rem Double-click to run. Keep this window open while testing.
cd /d %~dp0..
start "" http://localhost:9001
D:\anaconda3\envs\ml\python.exe app\agent.py
pause
