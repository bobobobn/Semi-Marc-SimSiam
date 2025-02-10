@echo off
setlocal enabledelayedexpansion

:: 设置模型路径前缀
set "CHECKPOINT_PATH=checkpoints/simsiamda"

:: 运行循环，从 0 到 8
for /L %%i in (0,1,8) do (
    set "CHECKPOINT_FILE=checkpoint_000%%i.pth.tar"
    echo Running train.py with !CHECKPOINT_FILE!
    python train.py --pretrained_model "%CHECKPOINT_PATH%/!CHECKPOINT_FILE!"
)

echo All checkpoints have been processed.
pause
