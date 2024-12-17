import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class CLSToken(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        BATCH_SIZE = x.shape[0]

        # repeat the parameter to accomodate the batch size, and then concatenate along columns
        # leads to shape (batch, p x p + 1, embedding_dim)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=BATCH_SIZE)

        concat_tokens = torch.cat([cls_tokens, x], dim=1)
        return concat_tokens


class PositionalEncoding(nn.Module):
    def __init__(self, img_size, patch_size, dim):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn((img_size // patch_size)**2+1, dim))

    def forward(self, x):
        return self.positional_encoding + x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, scaling=True):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.scaling = scaling

        self.queries_keys_values = nn.Linear(dim, 3*dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        splits = rearrange(self.queries_keys_values(
            x), 'b n (h d qkv) -> (qkv) b h n d', qkv=3, h=self.num_heads)
        queries, keys, values = splits[0], splits[1], splits[2]

        attention = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        attention = nn.functional.softmax(attention, dim=-1)
        if self.scaling:
            attention = attention / (self.embedding_dim**0.5)

        attention = self.dropout(attention)

        output = torch.einsum('bhad, bhdv -> bhav', attention, values)
        output = rearrange(output, 'b h a v -> b a (h v)')
        return self.projection(output)


class MLP(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        self.hidden_dim = int(expansion_factor * dim)
        self.fc1 = nn.Linear(dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, expansion_factor=4, scaling=True):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, dropout, scaling)
        self.mlp = MLP(dim, expansion_factor)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x + self.norm1(self.attention(x))
        return x + self.norm2(self.mlp(x))


class VanillaViT(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class DiffAttnViT(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class nViT(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class nDiffAttnViT(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass