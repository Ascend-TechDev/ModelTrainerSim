"""KDA (chunked SSM-style linear attention) layer for Kimi 10T.

Workbook rows I37-I51: q/k/v proj [h, n*hd] x3, q/k/v conv1d
[n*hd, conv_size] x3, A_log [n], dt_bias [n*hd], b_proj [h, n],
g_proj [h, n*hd], f_a [h, hd], f_b [hd, n*hd], o_norm [hd],
o_proj [n*hd, h].  Per-layer params = 443,740,384.

FLOPs (R-column, per mbs token): the three qkv projections, f_a, f_b,
b_proj, g_proj and o_proj — the conv/SSM state steps carry no GEMM
count in the workbook.  Activation tensors follow N37-N50 plus the
补档 rows N76-N80 (silu/L2norm/gate/output).
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown


class KDA(BaseModule):
    module_type = "KDA"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]          # h
        self.num_heads: int = config["num_heads"]              # n
        self.head_dim: int = config["head_dim"]                # hd
        self.conv_size: int = config["conv_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self):
        H, N, HD = self.hidden_size, self.num_heads, self.head_dim
        bf = self.dtype
        out = []
        for label in ("q_proj", "k_proj", "v_proj"):            # I37-39
            out.append(GroupedParamBreakdown(label, (H, N * HD), H * N * HD, bf,
                                             group="muon", shard="tp"))
        for label in ("q_conv1d", "k_conv1d", "v_conv1d"):     # I40-42
            out.append(GroupedParamBreakdown(label, (N * HD, self.conv_size),
                                             N * HD * self.conv_size, bf,
                                             group="muon", shard="tp"))
        out.extend([
            GroupedParamBreakdown("A_log", (N,), N, bf, group="muon", shard="tp"),       # I43
            GroupedParamBreakdown("f_a_proj", (H, HD), H * HD, bf, group="muon", shard="tp"),  # I44
            GroupedParamBreakdown("f_b_proj", (HD, N * HD), HD * N * HD, bf,
                                  group="muon", shard="tp"),                             # I45
            GroupedParamBreakdown("dt_bias", (N * HD,), N * HD, bf, group="muon", shard="tp"),  # I46
            GroupedParamBreakdown("b_proj", (H, N), H * N, bf, group="muon", shard="tp"),  # I47
            GroupedParamBreakdown("g_proj", (H, N * HD), H * N * HD, bf,
                                  group="muon", shard="tp"),                             # I48
            GroupedParamBreakdown("o_norm", (HD,), HD, bf, group="muon", shard="tp"),    # I49
            GroupedParamBreakdown("o_proj", (N * HD, H), N * HD * H, bf,
                                  group="muon", shard="tp"),                             # I50
        ])
        return out

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, N, HD = self.hidden_size, self.num_heads, self.head_dim
        qkv = 3 * 2 * B * S * H * N * HD                         # R37-39
        f_a = 2 * B * S * H * HD                                 # R44
        f_b = 2 * B * S * HD * N * HD                            # R45
        b_proj = 2 * B * S * H * N                               # R47
        g_proj = 2 * B * S * H * N * HD                          # R48
        o_proj = 2 * B * S * N * HD * H                          # R50
        return qkv + f_a + f_b + b_proj + g_proj + o_proj

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, N, HD = self.hidden_size, self.num_heads, self.head_dim
        a = self.bytes["activation"]
        qkv_out = 3 * B * S * N * HD * a                         # N37-39
        conv_out = 3 * B * S * N * HD * a                        # N40-42
        f_a_out = B * S * HD * a                                 # N44
        f_b_out = B * S * N * HD * a                             # N45
        b_out = B * S * N * a                                    # N47
        g_out = B * S * N * HD * a                               # N48
        norm_out = B * S * N * HD * a                            # N49
        o_out = B * S * H * a                                    # N50
        silu3 = B * S * N * HD * 3 * a                           # N76
        l2norm2 = B * S * N * HD * 2 * a                         # N77
        b_silu = B * S * N * a                                   # N78
        gated = B * S * N * HD * a                               # N79
        kda_out = B * S * N * HD * a                             # N80
        return (qkv_out + conv_out + f_a_out + f_b_out + b_out + g_out + norm_out
                + o_out + silu3 + l2norm2 + b_silu + gated + kda_out)
