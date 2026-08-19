from tokenizer import batch
from classes import GPTBackbone, LayerNorm, FeedForward, TransformerBlock
from GPT2 import GPT2
from tokenizer import tokenzier
import torch.nn as nn
import torch
from config import GPT_CONFIG_124M as cfg
from services import *

start_context = "United States of America"
encoded = tokenzier.encode(start_context)
print(encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor.shape)

model = GPT2(cfg)


model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=cfg["context_length"]
    )

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenzier.decode(out.squeeze(0).tolist())
print(decoded_text)