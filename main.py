from tokenizer import batch
from GPT import GPT2
import torch.nn as nn
import torch
from config import GPT_CONFIG_124M as cfg

torch.manual_seed(123)
model = GPT2(cfg)
logits = model(batch)
print(logits)