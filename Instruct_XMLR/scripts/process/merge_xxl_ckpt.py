from fairseq import checkpoint_utils
import torch


def merge_ckpt(para_block=8):
    
    state = None
    for rank in range(para_block):
        
        local_path = "../checkpoint_last-model_part-{}.pt".format(rank)
        with open(local_path, "rb") as f:
            ckpt_state = torch.load(f, map_location=torch.device("cpu"))

        if state is None:
            state = ckpt_state
            continue
        else:
            ckpt_state = ckpt_state['model']    
        
        for k in list(ckpt_state.keys()):
            print(rank, k, ckpt_state[k].size())

            if "embed_tokens" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "embed_positions" in k:
                continue
            elif "decoder.layers" in k and "q_proj.weight" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "decoder.layers" in k and "q_proj.bias" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "decoder.layers" in k and "k_proj.weight" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "decoder.layers" in k and "k_proj.bias" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "decoder.layers" in k and "v_proj.weight" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "decoder.layers" in k and "v_proj.bias" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "decoder.layers" in k and "out_proj.weight" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=-1)
            elif "decoder.layers" in k and "out_proj.bias" in k:
                continue
            elif "decoder.layers" in k and "fc1.weight" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "decoder.layers" in k and "fc1.bias" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "decoder.layers" in k and "fc2.weight" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=-1)
            elif "decoder.layers" in k and "fc2.bias" in k:
                continue
            elif "norm" in k:
                continue
            elif "lm_head.weight" in k: 
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "lm_head.bias" in k:
                continue
            elif "lm_head.dense.weight" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "lm_head.dense.bias" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            elif "embed_length.weight" in k:
                state['model'][k] = torch.cat([state['model'][k], ckpt_state[k]], dim=0)
            else:
                import pdb; pdb.set_trace()
    torch.save(state, "../checkpoint_last.pt")

merge_ckpt(para_block=8)
