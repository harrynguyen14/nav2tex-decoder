import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Trả về complex freqs: (max_seq_len, head_dim // 2) dạng complex64.
    Được cache ở cấp model, không phải parameter — không lưu vào state_dict.
    """
    assert head_dim % 2 == 0, "head_dim phải chẵn để dùng RoPE"
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos   = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, theta)
    return torch.polar(torch.ones_like(freqs), freqs)

def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    x     : (B, n_heads, T, head_dim)  — float
    freqs : (T, head_dim // 2)         — complex64  (được slice theo T thực tế)
    Trả về tensor cùng shape và dtype với x.
    """
    dtype = x.dtype
    B, H, T, D = x.shape
    x_c = x.float().reshape(B, H, T, D // 2, 2)
    x_c = torch.view_as_complex(x_c)
    f = freqs[:T].unsqueeze(0).unsqueeze(0)
    x_rot = x_c * f
    return torch.view_as_real(x_rot).reshape(B, H, T, D).to(dtype)

class SqueezeAttention(nn.Module):
    """
    Causal self-attention với:
      1. K/V bottleneck trên feature dimension (theo UniMERNet)
      2. RoPE thay thế absolute positional embedding

    Bottleneck path: x → Linear(d_model, d_sq) → Linear(d_sq, d_model) → reshape heads
    RoPE được apply lên Q và K sau khi reshape thành heads (KHÔNG apply lên V).
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.d_model % config.squeeze_ratio == 0, (
            f"d_model ({config.d_model}) phải chia hết cho squeeze_ratio ({config.squeeze_ratio})"
        )

        self.n_heads   = config.n_heads
        self.head_dim  = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        d_sq = config.d_model // config.squeeze_ratio

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.k_squeeze = nn.Linear(config.d_model, d_sq,           bias=False)
        self.k_expand  = nn.Linear(d_sq,           config.d_model, bias=False)
        self.v_squeeze = nn.Linear(config.d_model, d_sq,           bias=False)
        self.v_expand  = nn.Linear(d_sq,           config.d_model, bias=False)

        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_expand(self.k_squeeze(x)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_expand(self.v_squeeze(x)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)

        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype),
            diagonal=1,
        )

        if attention_mask is not None:
            pad_mask  = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_bias = causal.unsqueeze(0).expand(B, 1, T, T).clone()
            attn_bias = attn_bias.masked_fill(pad_mask, float("-inf"))
        else:
            attn_bias = causal

        drop = self.dropout_p if self.training else 0.0
        out  = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class CrossAttention(nn.Module):
    """
    Cross-attention để nhận encoder output (vision encoder) sau khi pretrain xong.
    Dùng zero-init gate để fine-tuning khởi động mượt:
      - Lúc bắt đầu fine-tune: gate ≈ 0 → tanh(0) = 0 → cross-attn không ảnh hưởng
      - Model tự học dần mức độ attend vào encoder output
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads   = config.n_heads
        self.head_dim  = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        self.q_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        S = encoder_output.size(1)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_output).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_output).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        drop = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        out = self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

        return torch.tanh(self.gate) * out

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1     = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2     = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1   = nn.LayerNorm(config.d_model)
        self.sa      = SqueezeAttention(config)
        self.norm2   = nn.LayerNorm(config.d_model)
        self.cross   = CrossAttention(config)
        self.norm3   = nn.LayerNorm(config.d_model)
        self.ffn     = FFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.sa(self.norm1(x), freqs, attention_mask))

        if encoder_output is not None:
            x = x + self.dropout(self.cross(self.norm2(x), encoder_output))

        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

class DecoderLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embed_drop  = nn.Dropout(config.dropout)

        self.layers   = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embed.weight

        head_dim = config.d_model // config.n_heads
        freqs    = _build_rope_freqs(head_dim, config.max_seq_len)
        self.register_buffer("rope_freqs", freqs, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.shape

        x = self.embed_drop(self.token_embed(input_ids))

        freqs = self.rope_freqs[:T]

        for layer in self.layers:
            x = layer(x, freqs=freqs, attention_mask=attention_mask, encoder_output=encoder_output)

        x      = self.norm_out(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: str = "cpu"):
        import argparse
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        ckpt = Path(checkpoint_dir)
        with open(ckpt / "config.json") as f:
            cfg_dict = json.load(f)
        config = argparse.Namespace(**cfg_dict)

        model = cls(config).to(device)
        sd = load_file(ckpt / "model.safetensors", device=device)
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        return model