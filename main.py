from tokenizer import batch
from GPT import GPT2, LayerNorm, FeedForward
import torch.nn as nn
import torch
from config import GPT_CONFIG_124M as cfg

ffn = FeedForward(cfg)
x = torch.rand(2, 3 , 768)
out = ffn(x)
print(out.shape)
print(out)