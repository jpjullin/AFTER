import torch.nn as nn
import torch


class SPE(torch.nn.Module):

    def __init__(self, dim=128, max_positions=10000, scale=32):
        super().__init__()
        self.embedding_size = dim
        self.max_positions = max_positions
        self.scale = scale

    def forward(self, x):
        x = x.view(-1)
        x = x * self.scale
        freqs = torch.arange(start=0,
                             end=self.embedding_size // 2,
                             dtype=torch.float32,
                             device=x.device)
        w = (1 / self.max_positions)**(2 * freqs / self.embedding_size)
        w = w[None, :]
        x = x[:, None]
        x = torch.cat([torch.sin(w * x), torch.cos(w * x)], dim=-1)
        return x.squeeze(1)
