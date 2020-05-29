#!/bin/sh

E=500
BS=32
LR=3e-4
LH=gaussian
DS=boneage
VS=2000
PT=20
WD=1e-7
GPU=0

#python -u train.py resnet50 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
#python -u train.py resnet101 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
#python -u train.py densenet121 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
python -u train.py densenet201 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
#python -u train.py efficientnetb0 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
#python -u train.py efficientnetb4 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
