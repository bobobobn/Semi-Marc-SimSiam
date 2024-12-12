@echo off
:: 设置输出文件路径
set OUTPUT_FILE=output.txt

:: 清空或新建输出文件
echo. > %OUTPUT_FILE%

:: 定义 excep_size 列表
set "EXCEP_SIZES=200 100 50 25 13"

:: 遍历列表并按顺序运行脚本
for %%s in (%EXCEP_SIZES%) do (
    echo Running with excep_size=%%s >> %OUTPUT_FILE%
    python .\main_simsiam.py --ssv_size=200 --normal_size=200 --excep_size=%%s >> %OUTPUT_FILE% 2>&1
    python .\main_lincls.py --ssv_size=200 --normal_size=200 --excep_size=%%s >> %OUTPUT_FILE% 2>&1
)
