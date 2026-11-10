import torch
import torch.nn as nn
import math
from einops import rearrange
from .nn_utils import silu, scaled_dot_product_attention

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=math.sqrt(2.0 / (in_features + out_features)), a=-3, b=3)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        result= x / rms * self.weight

        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(
        self,
        base: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.d_k = d_k

        freqs = 1.0 / (
            base ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        )

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * freqs[None, :]

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    
    def forward(
        self,
        x,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_pair = rearrange(
            x,
            "... (d_pair two) -> ... d_pair two",
            two=2
        )
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]

        x_even_new = x_even * cos - x_odd * sin
        x_odd_new = x_even * sin + x_odd * cos

        x_new = torch.stack([x_even_new, x_odd_new], dim = -1)
        return rearrange(x_new, "... d_pair two -> ... (d_pair two)")

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_k= d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor, # ... sequence_length d_model
        token_positions: torch.Tensor | None = None
    ):
        seq_len = x.size(-2)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        k = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",num_heads=self.num_heads, d_k=self.d_k)
        v = rearrange(v, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",num_heads=self.num_heads, d_k=self.d_k)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k= self.rope(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attn = scaled_dot_product_attention(q, k, v, mask=mask)

        attn = rearrange(attn, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")

        result = self.output_proj(attn)

        return result


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        d_k = d_model // num_heads
        rope = RoPE(rope_theta, d_k, context_length, device=device)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor
    ):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)