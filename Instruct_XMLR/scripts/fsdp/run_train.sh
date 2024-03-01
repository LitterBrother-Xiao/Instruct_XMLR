#!/bin/bash
src=\
tgt=\

export OMP_NUM_THREADS=\
export CUDA_VISIBLE_DEVICES=\
export CUDA_LAUNCH_BLOCKING=\

data_dir=[your preprocessed training data saved path]
save_dir=[your training checkpoint saved path]
xlmr_dir=[your pretrained xmlr saved path]
max_token=\

python ../../src/train_fsdp.py $data_dir \
    --user-dir ../../src \
    --ddp-backend fully_sharded \
    --fp16 \
    --checkpoint-activations \
    --truncate-source \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --no-reshard-after-forward \
    --no-save-optimizer-state \
    --task seq2seq_ft_task \
    --arch nar_xlmr_xl \
    --criterion cmlm_loss \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --lr-scheduler polynomial_decay --lr 2e-5 \
    --weight-decay 0.01 \
    --total-num-update 20000 --warmup-updates 500 \
    --max-epoch 50 \
    --no-progress-bar \
    --log-interval 50 \
    --save-dir $save_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $xlmr_dir \
    --length-predict