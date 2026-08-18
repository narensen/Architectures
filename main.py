from tokenizer import batch
from classes import GPTBackbone, LayerNorm, FeedForward, TransformerBlock
import torch.nn as nn
import torch
from config import GPT_CONFIG_124M as cfg

ffn = FeedForward(cfg)
x = torch.rand(2, 3 , 768)
block = TransformerBlock(cfg)
output = block(x)

print(x)