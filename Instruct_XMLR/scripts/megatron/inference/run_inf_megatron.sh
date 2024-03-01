# !/bin/bash
src=\
tgt=\
save_dir=[your training checkpoint saved path]
xlmr_dir=[your pretrained xmlr saved path]
bpe_dir=[your pretrained xmlr saved path]/sentencepiece.bpe.model
data_dir=[your evaluation data saved path]
world_size=1
export CUDA_LAUNCH_BLOCKING=\
export CUDA_VISIBLE_DEVICES=\
torchrun --master_port 29000 --nproc_per_node $world_size ../../../src/generate2.py $data_dir \
    --user-dir ../../../src \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --task seq2seq_ft_task \
    --arch nar_xlmr_xxl \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path ${save_dir} \
    --required-batch-size-multiple 1 \
    --batch-size 1 \
    --beam 1 \
    --iter-decode-max-iter 0 \
    --iter-decode-force-max-iter \
    --iter-decode-with-beam 1 \
    --memory-efficient-fp16 \
    --fp16 --fp16-init-scale 4 \
    --remove-bpe \
    
    


