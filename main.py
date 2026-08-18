from tokenizer import batch
from GPT import GPT2, LayerNorm
import torch.nn as nn
import torch
from config import GPT_CONFIG_124M as cfg

"""torch.manual_seed(123)
model = GPT2(cfg)
logits = model(batch)
print(logits)"""

batch = torch.randn(2,4)

ln = LayerNorm(emb_dim=4)
out_ln = ln(batch)
mean = out_ln.mean(dim=-1, keepdim = True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print(mean, var)