# 非专家 FSDP：全局 Ring

## 输入

- [模型输入](../model_inputs.md)：`P = 10^9 × layer_non_expert_params_b`，包含共享专家、Router、Engram 投影/门控，排除 Engram 查表权重。
- [基础场景](scenario_scan.md)：每个 SU 有 S 张卡，共 k 个 SU，`N = S × k`。
- [场景输入](../scenario_inputs.md)：FP8/BF16 权重分别为 1/2 byte，梯度通信为 FP32、4 byte。
- [层耗时](layer_compute.md)：`T_fwd`、`T_bwd`，内部单位为秒。

## 流量与带宽

N 张卡组成一个全局单 Ring，同一 SU 的 rank 连续排列。k>1 时，每个 SU 有 S−1 条域内环边、1 条跨域环边。每条环边传输 N−1 个分片，每片为完整张量的 1/N；不执行显式的两级 All-Gather / Reduce-Scatter。

```text
M_w = P × b_weight
M_g = P × b_gradient_comm
V_ring(M) = (N−1)/N × M
```

按各链路的**单卡最大发送流量**估计带宽。当前 S≥128，公式如下：

| 方向 | 单卡最大 SU 流量 | 单卡最大 SO 流量 | SU 带宽需求 | SO 带宽需求 |
|---|---|---|---|---|
| 前向 All-Gather | `(N−1)/N × M_w` | k>1：`(N−1)/N × M_w`；k=1：0 | `V_fwd_SU / T_fwd` | `V_fwd_SO / T_fwd` |
| 反向 Reduce-Scatter | `(N−1)/N × M_g` | k>1：`(N−1)/N × M_g`；k=1：0 | `V_bwd_SU / T_bwd` | `V_bwd_SO / T_bwd` |

- SU 与 SO 的最大值可能对应不同卡，不能相加作为同一卡的总流量。按链路分别统计时，一张卡的发送量和接收量也不必相等；收发不相加。
- k>1 时，平均 SO 流量为 `V_ring/S`，但出口卡负载为 `V_ring`，带宽需求采用后者。这里未采用多 Ring 轮换出口或其他负载均衡。
- 前向预取深度为 1，当前层与下一层均取所选最大 MoE 层；反向采用等层稳态窗口，不计尾部。SU/SO 小块传输理想重叠，不额外计入环步进延迟或调度开销。
- k=1 时退化为纯 SU 通信；若通用函数输入 S=1，则 SU 流量为 0。

Ring 分片系数沿用[公共方法](../common_methods.md)。输出为单卡有效带宽；计算先用 byte/s，换为 GB/s 时除以 10⁹。

## 输出

独立文件：[results/非专家FSDP_全局Ring.xlsx](../../../results/非专家FSDP_全局Ring.xlsx)。包含汇总页和 16 个模型页，保留全部基础场景；使用结果时筛选 HBM 合法标记为 1。

共 27 列，按以下顺序排列：基础场景及 HBM 21 列、层耗时 2 列、带宽需求 4 列。

| 字段 | 单位 |
|---|---|
| `layer_fwd_time`、`layer_bwd_time` | ms |
| `FSDP_fwd_su_bandwidth`、`FSDP_fwd_so_bandwidth` | GB/s |
| `FSDP_bwd_su_bandwidth`、`FSDP_bwd_so_bandwidth` | GB/s |

字段按文件中的通信方式解释。此表输出带宽需求；原分层通信表中的流量、分档耗时和掩盖率仍只对应分层通信。

代码：[non_expert_fsdp_global_ring.py](../../../scripts/su_so_modeling/non_expert_fsdp_global_ring.py)。独立刷新：

```bash
python3 scripts/generate_non_expert_fsdp_global_ring.py
```

## 校验

- 基础 21 列与场景总表一致，层耗时与独立层耗时表一致，模型页与汇总页一致。
- 由每条环边的 N−1 次分片发送独立复核带宽；k>1、S>1 时同一方向的 SU/SO 单卡最大带宽需求相等。
- k=1 时 SO 为 0，SU 结果等于分层通信；反向使用 FP32 梯度及反向窗口。
