import torch
from torch import nn
from torch.optim import Adam
from utils.mlm import MLM

# instantiate the language model

from reformer_pytorch import ReformerLM

transformer = ReformerLM(
    num_tokens = 20000,
    dim = 512,
    depth = 1,
    max_seq_len = 1024
)

# plugin the language model into the MLM trainer

trainer = MLM(
    transformer,
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
)

opt = Adam(trainer.parameters(), lr=3e-4)

# one training step (do this for many steps in a for loop, getting new `data` each time)

data = torch.randint(0, 20000, (2, 15))
data[0, -1] = 0
data[0, -2] = 0
data[0, -3] = 0
data[1, -1] = 0
data[1, -2] = 0

loss = trainer(data)
loss.backward()
opt.step()
opt.zero_grad()