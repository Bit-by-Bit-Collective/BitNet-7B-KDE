# scripts/models.py
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# Quantized building blocks
# ------------------------

class BitLinear(nn.Module):
    """
    Linear layer with ternary weights (-1/0/+1) via STE and learned per-output scales.
    Master weights stay full precision; quantization only affects forward pass.

    Args:
        in_features, out_features: usual linear dims
        bias: optional bias (kept full precision)
        thresh_factor: threshold = mean(|W|) * thresh_factor
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        thresh_factor: float = 0.7,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.thresh_factor = thresh_factor

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Per-output scale (column vector to broadcast over in_features)
        self.scale = nn.Parameter(torch.ones(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # Ternarize with adaptive threshold
        w_abs = w.abs()
        threshold = w_abs.mean() * self.thresh_factor

        # Quantized weights (detached), then STE
        wq_detached = torch.sign(w) * (w_abs > threshold).float()
        w_ternary = w + (wq_detached - w).detach()

        # Learned scale per output channel
        w_scaled = w_ternary * self.scale

        y = F.linear(x, w_scaled, self.bias)
        return y


class FakeQuantActivation(nn.Module):
    """
    Symmetric fake quantization for activations with STE.
    Set bits to 8 (or higher) to effectively no-op.

    clip_range is fixed to [-4, +4] which works well for most transformer activations.
    """
    def __init__(self, bits: int = 8, clip: float = 4.0):
        super().__init__()
        self.bits = int(bits)
        self.clip = float(clip)

    def set_bits(self, bits: int):
        self.bits = int(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits >= 8:
            return x

        x_clamped = torch.clamp(x, -self.clip, self.clip)
        levels = (2 ** self.bits) - 1
        scale = (2 * self.clip) / levels

        # STE: forward uses quantized, backward uses identity
        xq_detached = torch.round(x_clamped / scale) * scale
        xq = x + (xq_detached - x).detach()
        return xq


class RMSNorm(nn.Module):
    """
    RMSNorm: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use float32 for stability in norm
        norm = x.norm(p=2, dim=-1, keepdim=True, dtype=torch.float32)
        rms = norm * (x.shape[-1] ** -0.5)
        return (x / (rms + self.eps)) * self.weight


# ------------------------
# Transformer components
# ------------------------

class BitNetAttention(nn.Module):
    """
    Multi-head attention using BitLinear projections and fake-quant activations.
    """
    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim

        self.q_proj = BitLinear(dim, inner, bias=False)
        self.k_proj = BitLinear(dim, inner, bias=False)
        self.v_proj = BitLinear(dim, inner, bias=False)
        self.o_proj = BitLinear(inner, dim, bias=False)

        self.act_quant = FakeQuantActivation(8)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

        if attn_mask is not None:
            # attn_mask expected: [B, 1, T, T] broadcastable over H
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)  # [B, T, C]

        out = self.o_proj(out)
        return self.act_quant(out)


class BitNetMLP(nn.Module):
    """
    SwiGLU MLP with BitLinear layers and fake-quant activations.
    hidden_dim is typically 4 * dim.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = BitLinear(dim, hidden_dim, bias=False)
        self.up_proj = BitLinear(dim, hidden_dim, bias=False)
        self.down_proj = BitLinear(hidden_dim, dim, bias=False)
        self.act_quant = FakeQuantActivation(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        out = self.down_proj(hidden)
        return self.act_quant(out)


class BitNetBlock(nn.Module):
    """
    Pre-norm Transformer block using BitNetAttention and BitNetMLP.
    """
    def __init__(self, dim: int, n_heads: int, head_dim: int, mlp_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = BitNetAttention(dim, n_heads, head_dim)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = BitNetMLP(dim, mlp_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ------------------------
# Language Model wrapper
# ------------------------

class BitNetLM(nn.Module):
    """
    BitNet-style Transformer LM:
      - Token embeddings (full precision)
      - N x BitNetBlock (ternary weights + fake-quant activations)
      - Final RMSNorm + full-precision LM head

    Forward returns:
      - if labels is None: logits [B, T, V]
      - else: (loss, logits) with standard CE on next-token (ignore_index=-100)
    """
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, head_dim: int):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.embed_tokens = nn.Embedding(vocab_size, dim)

        mlp_dim = dim * 4
        self.blocks = nn.ModuleList([
            BitNetBlock(dim, n_heads, head_dim, mlp_dim)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)  # keep full precision

        # Global activation quantizer for embeddings/inputs
        self.global_act_quant = FakeQuantActivation(8)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- Quantization control ----
    def set_activation_bits(self, bits: int):
        """Set activation fake-quant bits globally and per block."""
        self.global_act_quant.set_bits(bits)
        for blk in self.blocks:
            blk.attn.act_quant.set_bits(bits)
            blk.mlp.act_quant.set_bits(bits)

    def get_activation_bits(self) -> int:
        return self.global_act_quant.bits

    # ---- Forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, T]
            attention_mask: [B, T] with 1 for real tokens and 0 for pad (optional)
            labels: [B, T] targets; when provided, returns (loss, logits)

        Masking:
            We build a causal mask and OR it with key-padding mask derived from attention_mask.
        """
        x = self.embed_tokens(input_ids)
        x = self.global_act_quant(x)

        B, T = input_ids.size(0), input_ids.size(1)

        # Causal mask: prevent attending to future tokens
        causal = torch.triu(
            torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1
        )  # [T, T]
        attn_mask: Optional[torch.Tensor] = causal.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]

        # Key padding mask: don't attend to padded positions
        if attention_mask is not None:
            # attention_mask: 1=keep, 0=pad -> convert to key padding mask
            kpm = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_mask = attn_mask | kpm  # broadcast over batch

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, V]

        if labels is None:
            return logits

        # Standard next-token CE on student logits (ignore_index = -100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss, logits


# ------------------------
# Configs & factories
# ------------------------

MINI_CONFIG: Dict[str, int] = {
    "dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "head_dim": 64,
}

FULL_7B_DRYRUN_CONFIG: Dict[str, int] = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "head_dim": 128,
}


def make_model(
    vocab_size: int,
    config: Dict[str, int],
    *,
    activation_bits: int = 8,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> BitNetLM:
    """
    Build a BitNetLM from a config dict {dim, n_layers, n_heads, head_dim}.
    """
    model = BitNetLM(
        vocab_size=vocab_size,
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        head_dim=config["head_dim"],
    )
    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    model.set_activation_bits(activation_bits)
    return model


def model_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_num_params_million(model: nn.Module) -> float:
    return model_num_params(model) / 1e6


# ------------------------
# Smoke test
# ------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    V = 50257
    B, T = 2, 16

    # Mini config sanity
    model = make_model(V, MINI_CONFIG, activation_bits=8)
    x = torch.randint(0, V, (B, T))
    attn_mask = torch.ones(B, T, dtype=torch.long)
    logits = model(x, attention_mask=attn_mask)  # [B, T, V]
    print("logits:", logits.shape, "params(M):", round(model_num_params_million(model), 2))

    # Labels path
    labels = torch.randint(0, V, (B, T))
    loss, logits = model(x, attention_mask=attn_mask, labels=labels)
    print("loss:", float(loss), "logits:", logits.shape)

    # Flip activations to 4-bit
    model.set_activation_bits(4)
    logits2 = model(x, attention_mask=attn_mask)
    print("activation bits:", model.get_activation_bits(), "ok:", logits2.shape == logits.shape)
