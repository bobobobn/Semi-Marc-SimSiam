@echo off
setlocal enabledelayedexpansion
set PYTHONPATH=%cd%
REM 定义参数列表
set sizes=16 32 64 128 256 500

REM 遍历参数并运行命令
for %%s in (%sizes%) do (
    echo Running with --batch-size %%s
    python .\main_byol.py --batch-size %%s
)

echo All tasks completed.