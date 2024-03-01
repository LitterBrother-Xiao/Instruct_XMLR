#!/bin/bash
src=\
tgt=\
save_dir=[your training checkpoint saved path]
xlmr_dir=[your pretrained xmlr saved path]
bpe_dir=[your pretrained xmlr saved path]/sentencepiece.bpe.model
data_dir=[your evaluation data saved path]
index=1
out_file_name=\
export CUDA_VISIBLE_DEVICES=\
python ../../../src/generate2.py ${data_dir} \
    --user-dir ../../../src \
    --task seq2seq_ft_task \
    --arch nar_xlmr_xl \
    --truncate-source \
    --max-source-positions 508 \
    --max-target-positions 508 \
    -s ${src} -t ${tgt} \
    --gen-subset test \
    --sacrebleu \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path $save_dir/checkpoint${index}.pt \
    --iter-decode-max-iter 9 \
    --iter-decode-with-beam 1 --remove-bpe \
    --iter-decode-force-max-iter \
    --batch-size 1 > ${save_dir}/${out_file_name}

cat ${save_dir}/${out_file_name} | grep -P "^D" | sort -V | cut -f 3- | sacremoses detokenize > ${save_dir}/out.hyp
cat ${save_dir}/${out_file_name} | grep -P "^T" | sort -V | cut -f 2- | sacremoses detokenize > ${save_dir}/out.ref

        
        
        
