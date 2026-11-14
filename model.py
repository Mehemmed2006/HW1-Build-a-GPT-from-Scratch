
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
    """Manual multi-head causal attention with KV cache."""

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
            torch.ones(
                config.block_size,
                config.block_size,
                dtype=torch.bool,
            )
        )
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len, channels = x.shape

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

        past_len = 0

        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)

            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        total_len = k.size(2)

        if total_len > self.causal_mask.size(0):
            raise ValueError("Sequence length exceeds block_size")

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        query_positions = torch.arange(
            past_len,
            past_len + seq_len,
            device=x.device,
        )
        key_positions = torch.arange(total_len, device=x.device)

        mask = (
            key_positions.unsqueeze(0)
            <= query_positions.unsqueeze(1)
        )

        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        y = weights @ v

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_len, channels)

        y = self.proj(y)
        y = self.resid_dropout(y)

        if use_cache:
            return y, (k, v)

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
    """Transformer block supporting Pre-LN and Post-LN."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.norm_type = config.norm_type

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        if self.norm_type == "pre":
            if use_cache:
                attn_output, present_kv = self.attn(
                    self.ln1(x),
                    past_kv=past_kv,
                    use_cache=True,
                )
            else:
                attn_output = self.attn(self.ln1(x))

            x = x + attn_output
            x = x + self.mlp(self.ln2(x))

        else:
            if use_cache:
                attn_output, present_kv = self.attn(
                    x,
                    past_kv=past_kv,
                    use_cache=True,
                )
            else:
                attn_output = self.attn(x)

            x = self.ln1(x + attn_output)
            x = self.ln2(x + self.mlp(x))

        if use_cache:
            return x, present_kv

        return x


class GPT(nn.Module):
    """Decoder-only GPT language model with optional KV cache."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        if config.use_positional_embeddings:
            self.position_embedding = nn.Embedding(
                config.block_size,
                config.n_embd,
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
            config.n_embd,
            config.vocab_size,
            bias=False,
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
        past_key_values: tuple | None = None,
        use_cache: bool = False,
    ):
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")

        batch_size, seq_len = idx.shape

        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layer:
                raise ValueError("Invalid number of cached layers")

            past_len = past_key_values[0][0].size(2)

            for past_k, past_v in past_key_values:
                if past_k.size(2) != past_len or past_v.size(2) != past_len:
                    raise ValueError("Inconsistent KV cache lengths")
        else:
            past_len = 0

        if past_len + seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        token_emb = self.token_embedding(idx)
        x = token_emb

        if self.position_embedding is not None:
            positions = torch.arange(
                past_len,
                past_len + seq_len,
                device=idx.device,
            )
            position_emb = self.position_embedding(positions)
            x = x + position_emb

        x = self.embedding_dropout(x)

        presents = []

        for layer_index, block in enumerate(self.blocks):
            layer_past = (
                past_key_values[layer_index]
                if past_key_values is not None
                else None
            )

            if use_cache:
                x, present_kv = block(
                    x,
                    past_kv=layer_past,
                    use_cache=True,
                )
                presents.append(present_kv)
            else:
                x = block(x, past_kv=layer_past)

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

        if use_cache:
            return logits, loss, tuple(presents)

        return logits, loss

    def num_params(self) -> int:
        """Count unique trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
