import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_hiddens = num_hiddens
        self.num_patches = int((self.img_size**2) / (self.patch_size**2))
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=num_hiddens,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionHead(nn.Module):
    def __init__(self, dim, head_dim, dropout=0.1):
        super().__init__()
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, head_dim, bias=False)
        self.k = nn.Linear(dim, head_dim, bias=False)
        self.v = nn.Linear(dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(dim, self.head_dim, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))# skip connection around attention
        x = x + self.ffn(self.norm2(x))# skip connection around FF
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.norm(x)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        cls_t = x[:, 0]
        out = self.norm(cls_t)
        out = self.linear(out)
        return out


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, num_heads, num_classes, depth=6, dropout=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_hiddens))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, num_hiddens))
        self.encoder = TransformerEncoder(num_hiddens, depth, num_heads, mlp_dim=num_hiddens * 4)
        self.classification_head = ClassificationHead(num_hiddens, num_classes)

    def forward(self, x):
        batch = x.shape[0]
        patches = self.patch_embedding(x)
        patches = patches.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = self.classification_head(x)
        return x
