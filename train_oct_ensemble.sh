#!/bin/sh

E=500
BS=16
LR=3e-4
LH=gaussian
DS=oct
VS=850
PT=20
WD=1e-7
GPU=1

python -u train_ensemble.py densenet201 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --num_ensemble=0 --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
python -u train_ensemble.py densenet201 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --num_ensemble=1 --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
python -u train_ensemble.py densenet201 ${LH} ${DS} --batch_size=${BS} --init_lr=${LR} --epochs=${E} --valid_size=${VS} --lr_patience=${PT} --weight_decay=${WD} --num_ensemble=2 --gpu=${GPU} | tee `date '+%Y-%m-%d_%H-%M-%S'`.log
