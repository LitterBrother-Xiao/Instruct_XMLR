# Are Bert Family Good Instruction Followers? A Study on Their Potential And Limitations
This is the source code for above [paper](https://openreview.net/pdf?id=x8VNtpCu1I) published on ICLR 2024.

We finetune [XML-R](https://arxiv.org/pdf/1911.02116.pdf) with a multilingual instruction dataset [xp3](https://arxiv.org/abs/2211.01786) to explore the effectiveness of the BERT family
for instruction following and zero-shot learning. 

🌟 Instruct-XMLR outperforms [Bloomz](https://arxiv.org/abs/2211.01786) on all evaluation tasks and achieves comparable performance with [mT0](https://arxiv.org/pdf/2110.08207) on most tasks.  
🌟 Instruct-XMLR possesses strong task and language generalization abilities.  
🌟 Instruct-XMLR can accelerate decoding due to its non-autoregressive generation manner. 

## Introduction
Language modeling at scale has proven very effective and brought unprecedented success to natural language models. 
Many typical representatives, especially decoder-only models, e.g., BLOOM and LLaMA, and encoder-decoder models, e.g., Flan-T5 and AlexaTM, have exhibited incredible instruction-following capabilities while keeping strong task completion ability. 
These large language models can achieve superior performance in various tasks and even yield emergent capabilities, e.g., reasoning and universal generalization. 
Though the above two paradigms are mainstream and well explored, the potential of the BERT family, which are encoder-only based models and have ever been one of the most representative pre-trained models, also deserves attention, at least should be discussed. 

## Install Requirements
To get started with this repository, you need to install required packages. Make sure you have Pytorch installed, then you can do this via pip using the following commands:
```
pip install fairseq
pip install fairscale
```

## Data Preparation
We use partial data from xp3 for instruction tuning in our paper.  
**Step1:** Download the dataset from [huggingface](https://huggingface.co/datasets/bigscience/xP3).  
**Step2:** Sample partial data from original xp3 but remaining strictly consistent in the composition ratio of each language, the number of tasks and prompt formats.  
**Step3:** Preprocess the data using Fairseq:
```
bash ../prepare_data.sh
```

## Model Training
We use fsdp to speed up training, please refer to **../src/fsdp** for more implementation details.

You can simply run the following script to begin training.
```
bash ../run_train.sh
```

If you don't have enough GPU memory, we also support [megatron-LM](https://github.com/NVIDIA/Megatron-LM), please refer to ** for more implementation details. 

## Model Inference
Make sure you have preprocess the evaluation data, then you can simply run the following script to begin inference.
```
bash ../run_inf.sh
```


## Any Questions?
If you have any questions related to this code, feel free to email us (ysxiaoo@stu.suda.edu.cn).
## Citation Details
If you use the Instruct-XMLR models for your research, please cite our paper via:
```
@inproceedings{
xiao2024are,
title={Are Bert Family Good Instruction Followers?  A Study on Their Potential And Limitations},
author={Yisheng Xiao and Zechen Sun and Juntao Li and Min Zhang and Zechang Li and Qingrong Xia and Xinyu Duan and Zhefeng Wang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=x8VNtpCu1I}
}
```

