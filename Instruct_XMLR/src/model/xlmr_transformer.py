# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple
import os
import math
import logging

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from fairseq import utils
from torch.nn import Linear
from fsdp.fully_sharded_data_parallel import fsdp_enable_wrap, fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.fairseq_dropout import FairseqDropout
import numpy as np
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
)
logger = logging.getLogger(__name__)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats

class XLMRTransformer(nn.Module):

    def __init__(self, cfg, src_dict, tgt_dict, embed_tokens):
        super().__init__()
        
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.embed_tokens = embed_tokens
        self.embed_dim = cfg.decoder_embed_dim
        self.num_layers = cfg.decoder_layers
        self.normalize_before = cfg.decoder_normalize_before

        self.pad = self.tgt_dict.pad()
        self.bos = self.tgt_dict.bos()
        self.eos = self.tgt_dict.eos()
        self.unk = self.tgt_dict.unk()
        
        self.embed_positions = (
            PositionalEmbedding(512,
                self.embed_dim,
                padding_idx=self.pad,
                learned=True,
            )
        )

        self.layers = torch.nn.ModuleList()
        self.layers.extend(
            [
                self.build_decoder_layer(cfg)
                for _ in range(self.num_layers)
            ]
        )
        self.layer_norm = LayerNorm(self.embed_dim)

        self.lm_head = XLMRHead(
            embed_dim=self.embed_dim,
            output_dim=len(self.tgt_dict),
            weight=self.embed_tokens.weight,
        )
        self.embed_length = Embedding(512, self.embed_dim, None)
        self.dropout_module = FairseqDropout(0.1)

    def forward_length(self, enc_feats, src_masks):
        enc_feats = _mean_pooling(enc_feats.transpose(0, 1), src_masks)
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1)
    
    def forward_length_prediction(self, length_out, tgt_tokens=None):
        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.pad).sum(1).long()
            length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=512)
        else:
            pred_lengs = length_out.max(-1)[1]
            length_tgt = pred_lengs
        return length_tgt

    def build_decoder_layer(self, cfg):
        layer = XLMRTransformerLayer(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def output_layer(self, x):
        return self.lm_head(x).float()


    def forward(self, source, target):
        
        src_embed = self.embed_tokens(source)
        src_x = src_embed + self.embed_positions(source)

        tgt_embed = self.embed_tokens(target)
        tgt_x = tgt_embed + self.embed_positions(target)

        if not self.normalize_before:
            src_x = self.layer_norm(src_x)
            tgt_x = self.layer_norm(tgt_x)

        src_x = self.dropout_module(src_x)
        tgt_x = self.dropout_module(tgt_x)
        src_key_padding_mask = source.eq(self.pad)
        tgt_key_padding_mask = target.eq(self.pad)
        
        hidden_state = [src_x]

        tgt_start_idx = src_x.size(1)
        x = torch.cat([src_x, tgt_x], dim=1)
        key_padding_mask = torch.cat([src_key_padding_mask, tgt_key_padding_mask], dim=1)
        
        # attention mask, i.e., similar to unilm, src tokens can not see tgt tokens
        tgt_len = tgt_x.size(1)
        src_len = src_x.size(1)
        self_attn_mask = torch.zeros([src_len + tgt_len, src_len + tgt_len]).to(src_x)
        self_attn_mask[:src_len,src_len:,] = float("-inf")

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                key_padding_mask,
                self_attn_mask
            )
            hidden_state.append(x[:,:tgt_start_idx,:])

        if self.normalize_before:
            src_x = self.layer_norm(x[:,:tgt_start_idx,:])
            tgt_x = self.layer_norm(x[:,tgt_start_idx:,:])
        
        return src_x, tgt_x, src_key_padding_mask, tgt_key_padding_mask, hidden_state

    def forward_enc(self, tokens):
        
        embed = self.embed_tokens(tokens)
        x = embed + self.embed_positions(tokens)

        if not self.normalize_before:
            x = self.layer_norm(x)

        x = self.dropout_module(x)
        key_padding_mask = tokens.eq(self.pad)
        
        hidden_state = [x]
        for i, layer in enumerate(self.layers):
            
            x = layer(
                x,
                key_padding_mask,
            )
            hidden_state.append(x)

        if self.normalize_before:
            x = self.layer_norm(x)
        
        return x, key_padding_mask, hidden_state

    def forward_dec(self, encoder_out, tokens):
        
        embed = self.embed_tokens(tokens)
        x = embed + self.embed_positions(tokens)

        if not self.normalize_before:
            x = self.layer_norm(x)

        x = self.dropout_module(x)
        tgt_key_padding_mask = tokens.eq(self.pad)
        
        hidden_state = [x]
        tgt_start_idx = encoder_out["encoder_padding_mask"][0].size(1)
        key_padding_mask = torch.cat([encoder_out["encoder_padding_mask"][0], tgt_key_padding_mask], dim=1)
        for i, layer in enumerate(self.layers):
            
            x_concat = torch.cat([encoder_out["encoder_states"][i], x], dim=1)
            
            x = layer(
                x_concat,
                key_padding_mask,
            )[:, tgt_start_idx:, :]
            hidden_state.append(x[:, tgt_start_idx:, :])

        if self.normalize_before:
            x = self.layer_norm(x)
        
        return x, key_padding_mask, hidden_state
    

class XLMRHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn("gelu")
        self.layer_norm = LayerNorm(embed_dim)
        
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class XLMRTransformerLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.embed_dim = cfg.decoder_embed_dim
        self.num_heads = cfg.decoder_attention_heads
        self.ffn_embed_dim = cfg.decoder_ffn_embed_dim

        self.self_attn = XLMRAttention(self.num_heads, self.embed_dim)

        self.activation_fn = utils.get_activation_fn("gelu")
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        
        self.normalize_before = cfg.decoder_normalize_before
        self.dropout_module = FairseqDropout(0.1)
        
    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask=None,
    ):
        
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x = residual + self.dropout_module(self.self_attn(x, key_padding_mask, attn_mask))
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        
        x = residual + self.dropout_module(self.fc2(self.activation_fn(self.fc1(x))))
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class XLMRAttention(nn.Module):

    def __init__(self, num_heads, embed_dim):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.dropout_module = FairseqDropout(0.1)

    def forward(self, query, key_padding_mask, attn_mask=None):
        
        bsz, src_len, embed_dim = query.size()
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(2, 3))

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(1)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )
        attn_softmax_scores = F.softmax(attn_scores.float(), dim=-1).type_as(q)
        output = torch.matmul(self.dropout_module(attn_softmax_scores), v)
        output = output.transpose(1, 2).contiguous().view(bsz, src_len, -1)
        return self.out_proj(output)

