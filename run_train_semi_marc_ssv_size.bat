@echo off

set base_path=D:\1D-CNN-for-CWRU-master\checkpoints\simsiam_ssv_size

@REM python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_size_0010.pth.tar" --ssv_size 10
@REM python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_size_0020.pth.tar" --ssv_size 20
@REM python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_size_0050.pth.tar" --ssv_size 50
@REM python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_size_0100.pth.tar" --ssv_size 100
python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_size_0150.pth.tar" --ssv_size 150
python .\train_semi_marc.py --pretrained_model "%base_path%\checkpoint_0799_size_0200.pth.tar" --ssv_size 200

echo All tasks completed.