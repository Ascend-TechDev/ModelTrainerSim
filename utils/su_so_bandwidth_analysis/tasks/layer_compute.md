# 单层计算耗时

状态：已实现。代码：[layer_compute.py](../../../scripts/su_so_modeling/layer_compute.py)。

## 输入

- [模型输入](../model_inputs.md)：最大 MoE 层的非专家参数、路由专家总参数、E、Top-K。
- [场景输入](../scenario_inputs.md)：单卡 token 数、有效算力、反向/前向耗时比。

## 方法

```text
P_non = 10^9 × layer_non_expert_params_b
P_expert = 10^9 × layer_routed_expert_total_params_b / E
t = micro_batch_size × seq_len

C_non_fwd = 2 × P_non × t
C_moe_fwd = 2 × P_expert × t × Top-K
T_fwd = (C_non_fwd + C_moe_fwd) / F_eff
T_bwd = backward_time_ratio × T_fwd
```

F_eff 使用 operations/s，已含计算利用率。非专家包含共享专家、Router、Engram 投影/门控，排除 Engram 查表权重。

层计算量固定采用上述激活参数量估计。Attention 投影参数计入非专家激活参数；不追加 QKᵀ/AV 等 Attention 算子开销，不计重计算开销。反向耗时按前向的 2 倍估算。

完全均衡时，每个 EP rank 计算 E/p 个专家，每专家处理 `p×t×Top-K/E` 次分配，所以单卡专家计算量为 `2×P_expert×t×Top-K`。不再除以 EP/FSDP/E-FSDP，也不重复乘 Top-K。

## 输出

`layer_times(model, effective_tops, tokens_per_card, backward_ratio)` 返回前向、反向耗时，单位秒。独立输出 [results/单层计算耗时.xlsx](../../../results/单层计算耗时.xlsx)：公共基础 21 列，加以下 2 列，共 23 列。

| 字段 | 单位 |
|---|---|
| `layer_fwd_time` | ms |
| `layer_bwd_time` | ms |

同一模型、精度下的层耗时相同。当前反向耗时为前向的 2 倍。FSDP 任务直接复用此计算函数，并在自己的结果表中保留相同的两列耗时。

## 校验与边界

独立计算非专家和专家两项 FLOPs，核查秒到毫秒的转换。Engram 查表访问耗时见 [待确认项](../open_issues.md)。
