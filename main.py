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
num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, numn_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="I love you", tokenizer=tokenzier, device="cuda",
)

print("-----------------------------------------------------------------------------------------")
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
model=model,
idx=text_to_token_ids("amazement", tokenizer),
max_new_tokens=25,
context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))