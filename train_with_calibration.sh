#!/bin/sh

BS=16
LR=3e-4
LH=gaussian
PT=20
WD=1e-7
GPU=1

#python -u train_with_calibration.py efficientnetb0 ${LH} endovis --batch_size=${BS} --init_lr=${LR} --epochs=200 --lr_patience=${PT} --weight_decay=${WD} --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
python -u train_with_calibration.py resnet50 ${LH} oct --batch_size=${BS} --init_lr=${LR} --epochs=500 --valid_size=850 --lr_patience=${PT} --weight_decay=${WD} --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
