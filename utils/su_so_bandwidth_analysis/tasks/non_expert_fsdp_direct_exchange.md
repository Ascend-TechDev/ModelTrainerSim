# 非专家 FSDP：各 Rank 直接交互

## 输入

- [模型输入](../model_inputs.md)：`P = 10^9 × layer_non_expert_params_b`，包含共享专家、Router、Engram 投影/门控，排除 Engram 查表权重。
- [基础场景](scenario_scan.md)：每个 SU 有 S 张卡，共 k 个 SU，`N = S × k`。
- [场景输入](../scenario_inputs.md)：FP8/BF16 权重分别为 1/2 byte，梯度通信为 FP32、4 byte。
- [层耗时](layer_compute.md)：`T_fwd`、`T_bwd`，内部单位为秒。

## 流量与带宽

每张卡直接与其余 N−1 张卡交换分片。域内走 SU，跨域直接走 SO，不额外计入经 SU 转发到出口的流量，也不使用跨 SU 的权重复用或预归约。

- All-Gather：向每个其他 rank 发送自己的 `M_w/N` 权重分片。
- Reduce-Scatter：向每个其他 rank 发送其最终负责的 `M_g/N` 梯度分片，由目标 rank 完成归约；不乘 All-Reduce 的系数 2。

```text
M_w = P × b_weight
M_g = P × b_gradient_comm
SU 通信对象数 = S−1
SO 通信对象数 = N−S
```

各卡负载均衡，按单卡单向发送量计算：

| 方向 | SU 流量 | SO 流量 | SU 带宽需求 | SO 带宽需求 |
|---|---|---|---|---|
| 前向 All-Gather | `(S−1)/N × M_w` | `(N−S)/N × M_w` | `V_fwd_SU / T_fwd` | `V_fwd_SO / T_fwd` |
| 反向 Reduce-Scatter | `(S−1)/N × M_g` | `(N−S)/N × M_g` | `V_bwd_SU / T_bwd` | `V_bwd_SO / T_bwd` |

相对于[分层通信](non_expert_fsdp.md)，同一方向的 SU 流量和带宽为其 1/k，SO 流量和带宽为其 S 倍。两种链路流量之和为 `(N−1)/N × M`。

前向预取深度为 1，当前层与下一层均取所选最大 MoE 层；反向采用等层稳态窗口，不计尾部。SU/SO 小块传输理想重叠，忽略消息启动延迟及其他通信争用。带宽采用有效带宽口径，不额外折减网络利用率。

## 输出

独立文件：[results/非专家FSDP_各Rank直接交互.xlsx](../../../results/非专家FSDP_各Rank直接交互.xlsx)。包含汇总页和 16 个模型页，保留全部基础场景；使用结果时筛选 HBM 合法标记为 1。

共 27 列，按以下顺序排列：基础场景及 HBM 21 列、层耗时 2 列、带宽需求 4 列。

| 字段 | 单位 |
|---|---|
| `layer_fwd_time`、`layer_bwd_time` | ms |
| `FSDP_fwd_su_bandwidth`、`FSDP_fwd_so_bandwidth` | GB/s |
| `FSDP_bwd_su_bandwidth`、`FSDP_bwd_so_bandwidth` | GB/s |

字段按文件中的通信方式解释。此表输出带宽需求；原分层通信表中的流量、分档耗时和掩盖率仍只对应分层通信。

代码：[non_expert_fsdp_direct_exchange.py](../../../scripts/su_so_modeling/non_expert_fsdp_direct_exchange.py)。独立刷新：

```bash
python3 scripts/generate_non_expert_fsdp_direct_exchange.py
```

## 校验

- 基础 21 列与场景总表一致，层耗时与独立层耗时表一致，模型页与汇总页一致。
- 按通信对象数和分片大小独立复核；两链路的带宽乘对应计算窗口，应等于单卡总发送量 `(N−1)/N × M`。
- k=1 时 SO 为 0，SU 结果等于分层通信；反向使用 FP32 梯度及反向窗口。
