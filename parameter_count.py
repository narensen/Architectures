from classes import GPTBackbone, LayerNorm, FeedForward, TransformerBlock
from Attentions.MultiHeadAttention import MultiHeadAttention
from GPT2 import GPT2
import torch.nn as nn
import torch
from config import GPT_CONFIG_124M as cfg


# weight tying (total model)
model = GPT2(cfg)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

total_params_gpt2 = (
total_params - sum(p.numel()
for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
f"considering weight tying: {total_params_gpt2:,}"
)

#parameter of fedforward and attention modules
ffn = FeedForward(cfg)
ffn_params = sum(p.numel() for p in ffn.parameters())
print(f"FFN PARAMS : {ffn_params}" )

#parameter of multi-head
multi_head = MultiHeadAttention(d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length= cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])

multi_head_params = sum(p.numel() for p in multi_head.parameters())
print(f"Multi-Head Params {multi_head_params}")