# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import torch.nn.functional as F
from fairseq.utils import new_arange
from fairscale.nn.model_parallel.cross_entropy import vocab_parallel_cross_entropy
from fairseq.utils import safe_getattr, safe_hasattr

@register_criterion("cmlm_loss")
class CMLMLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
    ):
        super().__init__(task)
        self.eps = 0.1
        self.pad = task.tgt_dict.pad()
        self.bos = task.tgt_dict.bos()
        self.eos = task.tgt_dict.eos()
        self.unk = task.tgt_dict.unk()
        self.mask = task.tgt_dict.index('<mask>')
        self.length_predict = task.length_predict

    def _random_mask(self, target_tokens):
        target_masks = (
            target_tokens.ne(self.pad) & target_tokens.ne(self.bos) & target_tokens.ne(self.eos)
        )
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.
        
        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        prev_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), self.mask
        )
        return prev_target_tokens
    
    def _fix_mask(self, target_tokens, ratio=0.15):
        pad = self.pad
        bos = self.bos
        eos = self.eos
        mask = self.mask
        target_masks = (target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos))
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * ratio
        # target_length = target_length + 1  # make sure to mask at least one token.
        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        target_mask = target_cutoff.scatter(1, target_rank, target_cutoff)
        mask_token = target_tokens.clone().masked_fill(target_mask, mask)
        return mask_token


    def forward(self, model, sample, reduce=True):
        source = sample["net_input"]["src_tokens"]
        source_mask = self._fix_mask(source)
        target = sample["target"]
        target_mask = self._random_mask(target)
        mask = (target != target_mask)
        output, length_out, length_tgt = model(source_mask, target_mask)
        loss, nll_loss = self.label_smooth_loss(output[mask], target[mask])
        
        length_loss = 0 
        pre_loss = loss
        if self.length_predict:
            length_loss = self.length_loss(length_out, length_tgt)
            loss += 0.1 * length_loss
        
        sample_size = 1
        logging_output = {
            "loss": pre_loss.data,
            "nll_loss": nll_loss.data,
            "length_loss": length_loss.data if length_loss else 0, 
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output
        
    def label_smooth_loss(self, net_out, net_target):
        net_logits = F.log_softmax(net_out, dim=-1)
        nll_loss = F.nll_loss(net_logits, net_target, reduction="none").float().mean()
        loss = nll_loss * (1. - self.eps) - net_logits.float().mean() * self.eps
        return loss, nll_loss

    def length_loss(self, length_out, length_tgt):
        length_logits = F.log_softmax(length_out, dim=-1)
        length_loss = F.nll_loss(length_logits, length_tgt, reduction="none").float().mean()
        return length_loss


    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        length_loss_sum = sum(log.get("length_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "length_loss", length_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
