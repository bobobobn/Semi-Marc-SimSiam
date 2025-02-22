@echo off
setlocal enabledelayedexpansion

REM 定义参数列表
set sizes=150 200

REM 遍历参数并运行命令
for %%s in (%sizes%) do (
    echo Running with --ssv_size %%s
    python .\main_simsiam.py --ssv_size %%s
)

echo All tasks completed.