@echo off
title FL-Server-8000
rem Start FL platform server on port 8000
rem Double-click to run. Keep this window open while testing.
cd /d %~dp0
D:\anoconda\envs\fl\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
pause
