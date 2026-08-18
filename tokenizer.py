import tiktoken
import torch

tokenzier = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenzier.encode(text1)))
batch.append(torch.tensor(tokenzier.encode(text2)))
batch = torch.stack(batch, dim=0)
print(batch)