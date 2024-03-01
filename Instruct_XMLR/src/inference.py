# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from model.xlmr_model import NARXLMR

import argparse
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(xlmr_model):
    
    # load from txt
    # prompts = [
    #     "给我讲一个曹操的故事.",
    #     "中国的最好的十所大学排名是什么？",
    # ]

    prompts = [
        "Who is the author of the Lord of the Rings",
        "Give me a story about Snow White",
        "Give me a story about cao cao.",
        "Write a short story in third person narration about a protagonist who has to make an important career decision.",
    ]


    # load from files
    # prompts = open("alpaca/scripts/assert/test.src").readlines()

    eval_kwargs = dict(sampling=True, sampling_topp=0.95, temperature=0.8)
    for prompt in prompts:
        print("-----" * 20)
        prompt_text = prompt
        print(prompt_text)
        output = xlmr_model.sample([prompt_text], **eval_kwargs)[0][0]
        print(output)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="",
        help="path containing model file",
    )
    parser.add_argument(
        "--model-file",
        default="",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--lora-model-inf",
        default="",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--lora-tuning",
        action="store_true",
        default=False,
        help="if true use XSUM_KWARGS else CNN_KWARGS",
    )

    parser.add_argument("--bpe",)
    parser.add_argument("--sentencepiece-model")
    args = parser.parse_args()
    
    kwargs = {
        "user_dir": "xlmr/src",
        "bpe": args.bpe,
        "sentencepiece_model": args.sentencepiece_model,
        "source_lang": 'src',
        "target_lang": 'tgt',
        "task": "seq2seq_ft_task",
        "iter_decode_max_iter": 9,
    }
    xlmr_model = NARXLMR.from_pretrained(
        model_name_or_path=args.model_dir,
        checkpoint_file=args.model_file,
        **kwargs,
    )
    xlmr_model = xlmr_model.eval()
    if torch.cuda.is_available():
        xlmr_model = xlmr_model.half().cuda()
    
    generate(xlmr_model)


if __name__ == "__main__":
    main()
