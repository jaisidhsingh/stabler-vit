import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads, layer_num):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scale_value = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_num)
        
        self.norm = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, x):
        queries = rearrange(self.q(x), "b n (h d q) -> b n (q h) d", h=self.num_heads, q=2, d=self.head_dim)
        queries = queries * self.scale_value

        keys = rearrange(self.k(x), "b n (h d k) -> b n (k h) d", h=self.num_heads, k=2, d=self.head_dim)
        v = rearrange(self.v(x), "b n (h d) -> b h n d", h=self.num_heads, d=2*self.head_dim)

        attention = torch.einsum("bnqd,bnkd->bnqk", queries, keys)
        attention = torch.nan_to_num(attention)
        attention = F.softmax(attention, dim=-1, dtype=torch.float32)

        lambda_1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_value = torch.exp(lambda_1) - torch.exp(lambda_2) + self.lambda_init

        attention = rearrange(attention, "b n (q h) (k a) -> q k b n h a", q=2, k=2, h=self.num_heads, a=self.num_heads)
        attention = attention[0, 0, ...] - lambda_value * attention[1, 1, ...]

        out = torch.einsum("bnah,bhnd->bnad", attention, v)
        out = self.norm(out)
        out = out * (1 - self.lambda_init)
        out = rearrange(out, "b n h d -> b n (h d)")
        out = self.output_projection(out)

        return out
