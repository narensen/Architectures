from tokenizer import batch
from classes import GPTBackbone, LayerNorm, FeedForward, TransformerBlock
from GPT2 import GPT2
import torch.nn as nn
import torch
from config import GPT_CONFIG_124M as cfg


model = GPT2(cfg)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")