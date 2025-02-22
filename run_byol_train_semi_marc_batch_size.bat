@echo off

set base_path=D:\1D-CNN-for-CWRU-master\checkpoints\byol
@REM python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_batchsize_0016.pth.tar"
@REM python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_batchsize_0032.pth.tar"
python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_batchsize_0064.pth.tar"
python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_batchsize_0128.pth.tar"
python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_batchsize_0256.pth.tar"
python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_batchsize_0500.pth.tar"

echo All tasks completed.