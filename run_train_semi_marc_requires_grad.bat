@echo off
echo Starting sequential execution of train_semi_marc.py commands...

REM 运行第一个命令
echo Running: python .\train_semi_marc.py --requires_grad --semi_requires_grad
python .\train_semi_marc.py --requires_grad --semi_requires_grad

REM 运行第二个命令
echo Running: python .\train_semi_marc.py --requires_grad
python .\train_semi_marc.py --requires_grad

REM 运行第三个命令
echo Running: python .\train_semi_marc.py --semi_requires_grad
python .\train_semi_marc.py --semi_requires_grad

echo All tasks completed.