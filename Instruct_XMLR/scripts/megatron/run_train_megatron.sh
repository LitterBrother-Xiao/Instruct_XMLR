#!/bin/bash
src=\
tgt=\



data_dir=[your preprocessed training data saved path]
xlmr_dir=[your splited pretrained xmlr saved path]
save_dir=[your training checkpoint saved path]
max_token=\
world_size=8
update_freq=8


export CUDA_VISIBLE_DEVICES=\
torchrun --master_port 29000 --nproc_per_node $world_size ../../src/train_megatron.py $data_dir \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --user-dir ../../src \
    --truncate-source \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 510 \
    --max-target-positions 510 \
    --memory-efficient-fp16 \
    --fp16 --fp16-init-scale 4 \
    --checkpoint-activations \
    --task seq2seq_ft_task \
    --megatron-model \
    --arch nar_xlmr_xxl \
    --criterion cmlm_loss \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --update-freq $update_freq \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --lr-scheduler polynomial_decay --lr 2e-5 \
    --weight-decay 0.01 \
    --total-num-update 100000 --warmup-updates 500 \
    --max-epoch 10 \
    --no-progress-bar \
    --log-interval 100 \
    --save-dir $save_dir \
    --length-predict \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $xlmr_dir \
    --save-interval 30000 \



