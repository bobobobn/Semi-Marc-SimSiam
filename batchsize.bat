@echo off

:: 切换到 simsiam 目录并运行相应的批处理脚本
cd simsiam
call simsiam_batchsize.bat
cd ..

:: 切换到 SimCLR 目录并运行相应的批处理脚本
cd SimCLR
call simclr_batchsize.bat
cd ..

:: 切换到 byol 目录并运行相应的批处理脚本
cd byol
call byol_batchsize.bat
cd ..

:: 返回上一级目录并执行批处理脚本
call run_train_semi_marc_batch_size.bat
call run_simclr_train_semi_marc_batch_size.bat
call run_byol_train_semi_marc_batch_size.bat