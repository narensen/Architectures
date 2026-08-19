from tokenizer import batch
from classes import GPTBackbone, LayerNorm, FeedForward, TransformerBlock
from GPT2 import GPT2
from tokenizer import tokenzier
import torch.nn as nn
import torch
from config import GPT_CONFIG_124M as cfg
from services import *
from load_verdict import *
from train import *

model = GPT2(cfg)
model.to("cuda")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0004, weight_decay=0.1
)
num_epochs = 1000

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device="cuda",
    numn_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="I love you", tokenizer=tokenzier
)