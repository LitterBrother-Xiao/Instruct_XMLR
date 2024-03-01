SRC=src
TGT=tgt

ori_DATA=[your sampled data saved path]
DATA=[your preprocessed data saved path]
SPM=../sentencepiece/build/src/spm_encode
XLMR_MODEL=[your pretrained xmlr saved path]/sentencepiece.bpe.model
XLMR_DICT=[your pretrained xmlr saved path]/dict.txt

${SPM} --model=${XLMR_MODEL} < ${ori_DATA}/train.src > ${DATA}/train.spm.src
${SPM} --model=${XLMR_MODEL} < ${ori_DATA}/train.tgt > ${DATA}/train.spm.tgt
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --validpref ${DATA}/valid.spm \
  --trainpref ${DATA}/train.spm \
  --testpref ${DATA}/test.spm \
  --destdir ${DATA}/databin \
  --srcdict ${XLMR_DICT} \
  --tgtdict ${XLMR_DICT} \
  --workers 16 \
