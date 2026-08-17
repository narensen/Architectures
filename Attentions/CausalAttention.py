import torch
import torch.nn as nn


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length):

        super().__init__()
        self.W_keys = nn.Linear(d_in, d_out, bias=False)
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_values = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_keys(x)
        queries = self.W_query(x)
        values = self.W_values(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens].unsqueeze(0), -torch.inf) # pyright: ignore[reportCallIssue]
        attn_weights = torch.softmax( attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


import torch


torch.manual_seed(123)
d_in, d_out = 8, 16
batch = torch.randn(2, 4, d_in)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length)
context_vecs = ca(batch)

print("context_vecs.shape:", context_vecs.shape)

        