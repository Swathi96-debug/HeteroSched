@echo off
title HeteroSched Local GPU Dashboard Launcher
echo ==================================================================
echo   Starting HeteroSched GPU Dashboard in WSL2 (Ubuntu 24.04)...
echo ==================================================================
wsl.exe -d Ubuntu-24.04 -- bash -c "cd /home/swathi/HeteroSched && ./scripts/launch_dashboard.sh %*"
pause
