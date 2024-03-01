from fairseq import checkpoint_utils
import torch


def read_ckpt():
    local_path = "/../parallel_8/model-model_part-1.pt"
    with open(local_path, "rb") as f:
        state_all = torch.load(f, map_location=torch.device("cpu"))
    import pdb; pdb.set_trace()

def split_ckpt(para_block=8):
    
    for rank in range(para_block):
        
        local_path = "../checkpoint_last.pt"
        with open(local_path, "rb") as f:
            state = torch.load(f, map_location=torch.device("cpu"))

        vocab_size = 250880
        embed_size = 2560
        fc_size = 2560 * 4
        para_vocab_size = vocab_size // para_block
        para_embed_size = embed_size // para_block
        para_fc_size = fc_size // para_block
        
        print(para_vocab_size, para_embed_size, para_fc_size)
        
        # rescale vocab embedding:
        embed_weight = state['model']['encoder.sentence_encoder.embed_tokens.weight']
        temp_size = embed_weight.size(0)
        embed_weight_copy = torch.zeros(vocab_size, embed_size).type_as(embed_weight)
        embed_weight_copy[:temp_size, :] = embed_weight
        embed_weight = embed_weight_copy
        print(embed_weight.size())

        # rescale output mapping bias:
        embed_bias = state['model']['encoder.lm_head.bias']
        embed_bias_copy = torch.zeros(vocab_size).type_as(embed_weight)
        embed_bias_copy[:temp_size] = embed_bias
        embed_bias = embed_bias_copy
        print(embed_bias.size())

        for k in list(state['model'].keys()):
            
            if "layer_norm" in k or "embed_positions" in k or "version" in k or "layernorm_embedding" in k:
                print(k)
                state['model'][k] = state['model'][k].clone()
                continue

            if "encoder.sentence_encoder.embed_tokens.weight" in k:
                start_dim = rank * para_vocab_size
                end_dim = (rank + 1) * para_vocab_size
                print("convert {} from {} to {} ".format(k, start_dim, end_dim))
                state['model'][k] = embed_weight[start_dim:end_dim, :].clone()
            elif "encoder.sentence_encoder.layers" in k:
                if "fc" in k:
                    start_dim = rank * para_fc_size
                    end_dim = (rank + 1) * para_fc_size
                    print("convert {} from {} to {} ".format(k, start_dim, end_dim))
                    if "fc1.weight" in k:
                        state['model'][k] = state['model'][k][start_dim:end_dim, :].clone()
                    elif "fc1.bias" in k:
                        state['model'][k] = state['model'][k][start_dim:end_dim].clone()
                    elif "fc2.weight" in k:
                        state['model'][k] = state['model'][k][:, start_dim:end_dim].clone()
                else:
                    start_dim = rank * para_embed_size
                    end_dim = (rank + 1) * para_embed_size
                    print("convert {} from {} to {} ".format(k, start_dim, end_dim))
                    if "self_attn.out_proj.weight" in k:
                        state['model'][k] = state['model'][k][:, start_dim:end_dim].clone()
                    elif "self_attn.out_proj.bias" in k:
                        state['model'][k] = state['model'][k].clone()
                    else:
                        if "self_attn.q_proj.weight" in k:
                            state['model'][k] = state['model'][k][start_dim:end_dim, :].clone()
                        elif "self_attn.q_proj.bias" in k:
                            state['model'][k] = state['model'][k][start_dim:end_dim].clone()
                        elif "self_attn.k_proj.weight" in k:
                            state['model'][k] = state['model'][k][start_dim:end_dim, :].clone()
                        elif "self_attn.k_proj.bias" in k:
                            state['model'][k] = state['model'][k][start_dim:end_dim].clone()
                        elif "self_attn.v_proj.weight" in k:
                            state['model'][k] = state['model'][k][start_dim:end_dim, :].clone()
                        elif "self_attn.v_proj.bias" in k:
                            state['model'][k] = state['model'][k][start_dim:end_dim].clone()
                        else:
                            print(state['model'][k].size())
                            print(k)
                            raise NotImplementedError
            elif "lm_head.weight" in k:
                start_dim = rank * para_vocab_size
                end_dim = (rank + 1) * para_vocab_size
                state['model'][k] = embed_weight[start_dim:end_dim, :].clone()
            elif "lm_head.bias" in k:
                state['model'][k] = embed_bias.clone()
            elif "lm_head.dense.weight" in k:
                start_dim = rank * para_embed_size
                end_dim = (rank + 1) * para_embed_size
                print("convert {} from {} to {} ".format(k, start_dim, end_dim))
                state['model'][k] = state['model'][k][start_dim:end_dim, :].clone()
            elif "lm_head.dense.bias" in k:
                start_dim = rank * para_embed_size
                end_dim = (rank + 1) * para_embed_size
                print("convert {} from {} to {} ".format(k, start_dim, end_dim))
                state['model'][k] = state['model'][k][start_dim:end_dim].clone()
            else:
                print(state['model'][k].size())
                print(k)
                raise NotImplementedError
        torch.save(state, "../checkpoint_last-model_part-{}.pt".format(rank))


split_ckpt(para_block=8)

# read_ckpt()
