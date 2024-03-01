# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from omegaconf import II
import math
import logging

import torch 
from torch import Tensor, nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP
from .hub_interface import XLMRHubInterface
from .xlmr_transformer import XLMRTransformer
from .xlmr_megatron import XLMRMegatron
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq.utils import new_arange
from generator.iterative_refinement_generator import DecoderOut
from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel.layers import ParallelEmbedding, VocabParallelEmbedding
from sentencepiece import SentencePieceProcessor

logger = logging.getLogger(__name__)


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@dataclass
class XLMRConfig(FairseqDataclass):

    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )

    decoder_embed_dim: int = field(
        default=4096, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=16384, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=48, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=32, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=True, metadata={"help": "norm before"}
    )
    
    max_target_positions: Optional[int] = II("task.max_target_positions")
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": ("minimum number of params for a layer to be wrapped with FSDP()")
        },
    )
    

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

@register_model("nar_xlmr", dataclass=XLMRConfig)
class NARXLMR(BaseFairseqModel):
    
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.mask_idx = self.decoder.tgt_dict.index('<mask>')
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        xlmr_base(args)
        
        logger.info("rescale [src] dictionary: {} types and [tgt] dictionary: {} types".format(
            len(task.source_dictionary), len(task.target_dictionary)))
        
        
        if safe_getattr(task, "megatron_model", True):
            cls.initialize_model_parallel()
            
            task.source_dictionary.pad_to_multiple_(torch.distributed.get_world_size() * 8)
            task.target_dictionary.pad_to_multiple_(torch.distributed.get_world_size() * 8)
            
            embed_tokens = cls.build_megatron_embedding(args, task.target_dictionary, args.decoder_embed_dim)
            decoder = XLMRMegatron(
                args,
                task.source_dictionary,
                task.target_dictionary,
                embed_tokens
            )
        else:
            embed_tokens = cls.build_embedding(args, task.source_dictionary, args.decoder_embed_dim)
            decoder = XLMRTransformer(
                args,
                task.source_dictionary,
                task.target_dictionary,
                embed_tokens
            )
            
        return cls(decoder)
        
    @classmethod
    def initialize_model_parallel(cls):
        logger.info("llama model init process group")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        if not mpu.model_parallel_is_initialized():
            ws = torch.distributed.get_world_size()
            mpu.initialize_model_parallel(ws)

    @classmethod
    def build_megatron_embedding(cls, args, dictionary, embed_dim):
        return VocabParallelEmbedding(len(dictionary), embed_dim, init_method=lambda x: x)
        
    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim):
        return Embedding(len(dictionary), embed_dim, dictionary.pad())

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file,
        **kwargs
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            **kwargs,
        )
        return XLMRHubInterface(x["args"], x["task"], x["models"][0])

    def forward(self, source, target_mask):
        
        src_x, tgt_x, src_key_padding_mask, tgt_key_padding_mask, hidden_state = self.decoder(source, target_mask)
        tgt_out = self.decoder.output_layer(tgt_x)
        length_out = self.decoder.forward_length(src_x, src_key_padding_mask)
        length_tgt = self.decoder.forward_length_prediction(length_out, target_mask)
        return tgt_out, length_out, length_tgt

    def forward_encoder(self, source):
        
        src_x, src_padding, src_hiddens = self.decoder.forward_enc(source)
        return {
            "encoder_out": [src_x],
            "encoder_padding_mask": [src_padding],
            "encoder_states": src_hiddens,
            "src_tokens": [source],
        }

    def nucleus_sampling(self, probs, output_tokens, step, max_step):
        
        nucleus_p = 0.9
        temperature = (1.0 - step / max_step) * 2.0
        probs = F.softmax(probs / temperature, dim=-1)
        raw_indices_buf = probs.max(-1)[1].unsqueeze(-1)
        
        if nucleus_p > 0:
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum_probs = sorted_probs.cumsum(dim=2)
            mask = cumsum_probs.lt(nucleus_p)

            cumsum_mask = mask.cumsum(dim=2)
            last_included = cumsum_mask[:, :, -1:]
            last_included.clamp_(0, mask.size()[2] - 1)
            mask = mask.scatter_(2, last_included, 1)
            
            max_dim = last_included.max()
            truncated_mask = mask[:, :, : max_dim + 1]
            truncated_probs = sorted_probs[:, :, : max_dim + 1]
            truncated_indices = sorted_indices[:, :, : max_dim + 1]
            trimed_probs = truncated_probs.masked_fill_(~truncated_mask, 0)
        else:
            trimed_probs, truncated_indices = probs.topk(nucleus_k)
        
        bsz, seq_len, _ = trimed_probs.size()
        select_buf = torch.multinomial(trimed_probs.view(bsz * seq_len, -1), 1, replacement=True).view(bsz, seq_len)
        scores_buf = torch.gather(trimed_probs, dim=2, index=select_buf.unsqueeze(-1))
        indices_buf = torch.gather(truncated_indices, dim=2, index=select_buf.unsqueeze(-1))
        return torch.log(scores_buf), indices_buf

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        
        output_masks = output_tokens.eq(self.mask_idx)
        tgt_x, tgt_padding_mask, _ = self.decoder.forward_dec(encoder_out, output_tokens)
        tgt_out = self.decoder.output_layer(tgt_x)

        # _scores, _tokens = self.nucleus_sampling(tgt_out, output_tokens, step, max_step)
        _scores, _tokens = F.log_softmax(tgt_out, -1).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.decoder.tgt_dict.pad()), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.mask_idx)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None):
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(encoder_out["encoder_out"][0], encoder_out["encoder_padding_mask"][0]),
            tgt_tokens=tgt_tokens,
        )
        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)
        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.decoder.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.mask_idx
        )
        initial_output_tokens[:, 0] = self.decoder.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.decoder.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0]).float()

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )
    
    def initialize_output_tokens_span(self, encoder_out, prefix_tokens, span_length, bos_flag=True, eos_flag=False):
        initial_output_tokens = prefix_tokens.new_zeros(
            prefix_tokens.size(0), span_length
        ).fill_(self.mask_idx)
        if bos_flag:
            initial_output_tokens[:, 0] = self.decoder.bos
        if eos_flag:
            initial_output_tokens[:, -1] = self.decoder.eos
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0]).float()
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )



    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(0, new_order)]
        
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(0, new_order)

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]
        
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
        }
        
    def upgrade_state_dict_named(self, state_dict, name):
        
        for k in list(state_dict.keys()):
                        
            if "version" in k:
                del state_dict[k]
                continue

            if "encoder.sentence_encoder" in k:
                new_k = k.replace("encoder.sentence_encoder", "decoder")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

            if "encoder.lm_head" in k:
                new_k = k.replace("encoder.lm_head", "decoder.lm_head")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]   

        if "decoder.embed_length.weight" not in state_dict:
            state_dict["decoder.embed_length.weight"] = self.decoder.embed_length.weight

        super().upgrade_state_dict_named(state_dict, name)

def xlmr_base_architecture(args):

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 768 * 4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)

    args.max_source_positions = safe_getattr(args, "max_source_positions", 512)
    args.max_target_positions = safe_getattr(args, "max_target_positions", 512)


def xlmr_xl_architecture(args):

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2560)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 2560 * 4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 36)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)

    args.max_source_positions = safe_getattr(args, "max_source_positions", 512)
    args.max_target_positions = safe_getattr(args, "max_target_positions", 512)


def xlmr_xxl_architecture(args):

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 4096)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096 * 4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)

    args.max_source_positions = safe_getattr(args, "max_source_positions", 512)
    args.max_target_positions = safe_getattr(args, "max_target_positions", 512)

@register_model_architecture("nar_xlmr", "nar_xlmr_base")
def xlmr_base(args):
    xlmr_base_architecture(args)

@register_model_architecture("nar_xlmr", "nar_xlmr_xl")
def xlmr_xl(args):
    xlmr_xl_architecture(args)

@register_model_architecture("nar_xlmr", "nar_xlmr_xxl")
def xlmr_xxl(args):
    xlmr_xxl_architecture(args)