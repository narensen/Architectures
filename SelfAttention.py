import torch.nn as nn
import torch


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out):

        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec