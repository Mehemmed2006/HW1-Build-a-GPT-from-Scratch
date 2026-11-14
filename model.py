
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    use_positional_embeddings: bool = True
    norm_type: str = "pre"

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.norm_type not in ("pre", "post"):
            raise ValueError("norm_type must be 'pre' or 'post'")


class CausalSelfAttention(nn.Module):
    """Manual multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        if seq_len > self.causal_mask.size(0):
            raise ValueError("Sequence length exceeds block_size")

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        y = weights @ v

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_len, channels)

        y = self.proj(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Supports Pre-LN and Post-LN Transformer blocks."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.norm_type = config.norm_type

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "pre":
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.mlp(x))

        return x


class GPT(nn.Module):
    """Decoder-only GPT language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )

        if config.use_positional_embeddings:
            self.position_embedding = nn.Embedding(
                config.block_size, config.n_embd
            )
        else:
            self.position_embedding = None

        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        self.final_ln = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")

        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        token_emb = self.token_embedding(idx)
        x = token_emb

        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=idx.device)
            position_emb = self.position_embedding(positions)
            x = x + position_emb

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx")

            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    def num_params(self) -> int:
        """Count unique trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
