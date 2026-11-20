# 非专家 FSDP 流量与带宽

状态：已实现带宽需求、分档通信耗时、分项及整体掩盖率。代码：[non_expert_fsdp.py](../../../scripts/su_so_modeling/non_expert_fsdp.py)。

## 输入

- [模型输入](../model_inputs.md)：`P = 10^9 × layer_non_expert_params_b`，包含共享专家、Router、Engram 投影/门控，排除 Engram 查表权重。
- [基础场景](scenario_scan.md)：S、k，N=S×k。
- [场景输入](../scenario_inputs.md)：权重和梯度通信字节数。
- [层耗时](layer_compute.md)：T_fwd、T_bwd，单位秒。

掩盖率扩展使用以下链路带宽档位：

| 链路 | 符号 | 档位（GB/s） |
|---|---|---|
| SU | B_SU | 400、600、800、1000、1200、1600 |
| SO | B_SO | 100、200 |

档位直接作为单卡单向可用带宽，不额外折减网络利用率。每个档位同时用于该链路的前向和反向计算。它们是本任务的扩展输入，按列记录，不增加公共场景扫描维度。

代码中的 `SU_LINK_BANDWIDTHS_GBPS`、`SO_LINK_BANDWIDTHS_GBPS` 统一定义档位，计算、字段生成及校验共用。工作簿属性单独记录这两组输入。

## 流量模型

```text
M_w = P × b_weight
M_g = P × b_gradient_comm
```

每卡坐标为 `(a,u)`：a 为 SU 域编号，u 为域内 rank。张量分为 N 片，第 `u×k+a` 片归 `(a,u)`。SU 组固定 a、大小 S；SO 组固定 u、大小 k。SO Size 仍表示总卡数 N。

| 方向 | 逻辑分层顺序 | 每卡张量大小变化 | 每卡单向流量 |
|---|---|---|---|
| 前向 All-Gather | 先 SO | `M_w/N → M_w/S` | `V_fwd_SO = (k-1)/(S×k) × M_w` |
| 前向 All-Gather | 后 SU | `M_w/S → M_w` | `V_fwd_SU = (S-1)/S × M_w` |
| 反向 Reduce-Scatter | 先 SU | `M_g → M_g/S` | `V_bwd_SU = (S-1)/S × M_g` |
| 反向 Reduce-Scatter | 后 SO | `M_g/S → M_g/N` | `V_bwd_SO = (k-1)/(S×k) × M_g` |

梯度通信采用 FP32；不会把 BF16 梯度存储精度或四类训练状态字节数代入。收发不相加，Reduce-Scatter 不乘 All-Reduce 的系数 2。

## 前向掩盖假设

- 预取深度为 1：当前层计算时，预取下一层的非专家权重。
- 当前层与下一层均取输入表中选定的参数量最大 MoE 层，因此掩盖窗口为当前层的 `T_fwd`，预取流量为同一最大层的前向流量。
- 按小数据块流水传输，SU 与 SO 的传输相互掩盖；忽略两级先后顺序造成的耗时影响和具体调度时序。
- 先估计完全掩盖所需有效带宽，再按给定链路档位分别计算掩盖率；首层启动不计入本次层间掩盖估算。

逻辑分层用于计算流量。SU/SO 各自使用完整的 `T_fwd` 作为掩盖窗口，四个通信分项分别输出掩盖率。

## 带宽模型

```text
BW_fwd_SU = V_fwd_SU / T_fwd
BW_fwd_SO = V_fwd_SO / T_fwd
BW_bwd_SU = V_bwd_SU / T_bwd
BW_bwd_SO = V_bwd_SO / T_bwd
```

前向两项分别为上述假设下完全掩盖所需的最低有效带宽，SU/SO 共用完整的 `T_fwd` 窗口。现有 `V/T_fwd` 公式及表格数值保持不变。

反向采用等层近似，各层均取所选最大 MoE 层，允许跨层 SU/SO 流水。总流量与总计算时间之比为 `(L×V)/(L×T_bwd)=V/T_bwd`；不累加相互重叠的剩余窗口。暂不计通信尾部、启动延迟和其他通信争用。

## 分档通信耗时与掩盖率

令 X 为 SU 或 SO，B_X 为对应链路档位。V 采用 GB，B 采用 GB/s，以下 T 均为秒：

```text
T_comm_fwd_X(B_X) = V_fwd_X / B_X
T_comm_bwd_X(B_X) = V_bwd_X / B_X

R_fwd_X(B_X) = min(1, T_fwd / T_comm_fwd_X(B_X))
R_bwd_X(B_X) = min(1, T_bwd / T_comm_bwd_X(B_X))
```

反向分母使用反向通信耗时 `T_comm_bwd_X`，对应 FP32 梯度通信量。前向和反向、SU 和 SO 分别计算。

使用当前表格字段时：

```text
通信耗时_ms = 1000 × 对应流量_GB / 链路带宽_GBps
掩盖率 = min(1, 对应层耗时_ms / 通信耗时_ms)
       = min(1, 链路带宽_GBps / 对应带宽需求_GBps)  # 流量 > 0
```

- 掩盖率表示可被对应计算窗口覆盖的通信耗时比例。反向沿用稳态估计，不计通信尾部；该比例不是整步训练加速比。
- 流量为 0 时，通信耗时为 0，约定掩盖率为 1；因此 k=1 时各 SO 档位均记录 0 ms、100%，避免除零。
- 掩盖率保存为 0～1，Excel 显示为百分比；时间统一保存为 ms。判断是否达到 100% 使用未按显示精度舍入的值。
- 所有带宽档位必须大于 0。计算仍覆盖全部场景，使用结果时按 HBM 合法标记筛选。

## All-Gather 与 Reduce-Scatter 整体掩盖率

每个 SU/SO 带宽组合分别计算两个整体指标。沿用前向理想重叠、反向稳态流水假设，整体由掩盖率较低的一侧决定：

```text
R_AllGather(B_SU, B_SO) = min(R_fwd_SU(B_SU), R_fwd_SO(B_SO))
R_ReduceScatter(B_SU, B_SO) = min(R_bwd_SU(B_SU), R_bwd_SO(B_SO))
```

共 6×2=12 个带宽组合，输出 24 项整体掩盖率。k=1 时 SO 掩盖率为 100%，整体掩盖率等于对应 SU 掩盖率。反向不计通信尾部，整体指标仍为当前掩盖假设下的估计。

## 输出

独立输出 [results/非专家FSDP.xlsx](../../../results/非专家FSDP.xlsx)，共 87 列，覆盖全部场景，包括 HBM 不合法行：

| 字段组 | 列数 |
|---|---:|
| 公共基础 | 21 |
| 原有流量与带宽需求 | 8 |
| 层前向、反向耗时 | 2 |
| 分档通信耗时 | 16 |
| 分档分项掩盖率 | 16 |
| 12 个组合的整体掩盖率 | 24 |

`fsdp_metrics(...)` 保留原有以下 8 项结果；`fsdp_overlap_metrics(...)` 计算新增 56 项。原有 31 列的位置及数值保持不变。

| 字段 | 单位 |
|---|---|
| `FSDP_fwd_su_vol` | GB |
| `FSDP_fwd_so_vol` | GB |
| `FSDP_fwd_su_bandwidth` | GB/s |
| `FSDP_fwd_so_bandwidth` | GB/s |
| `FSDP_bwd_su_vol` | GB |
| `FSDP_bwd_so_vol` | GB |
| `FSDP_bwd_su_bandwidth` | GB/s |
| `FSDP_bwd_so_bandwidth` | GB/s |

带宽为单卡有效带宽；如需物理带宽，再除以网络有效利用率。模型存储仍用整模型参数，等层近似不改变 HBM 结果。

## 新增字段

保留原有 31 列及顺序，在末尾追加分档结果。每个场景仍占一行，页签布局和基础 21 列不变。分项指标按本链路档位计算；整体指标按 12 个 SU/SO 带宽组合增加列。

以下 `{B}` 替换为数值档位，`GBps` 为固定后缀。例如 `FSDP_fwd_su_comm_time_400GBps` 和 `FSDP_fwd_su_overlap_ratio_400GBps`。

| 分项 | B（GB/s） | 通信耗时字段（ms） | 掩盖率字段（0～1） | 新增列数 |
|---|---|---|---|---:|
| SU 前向 | 400、600、800、1000、1200、1600 | `FSDP_fwd_su_comm_time_{B}GBps` | `FSDP_fwd_su_overlap_ratio_{B}GBps` | 12 |
| SU 反向 | 400、600、800、1000、1200、1600 | `FSDP_bwd_su_comm_time_{B}GBps` | `FSDP_bwd_su_overlap_ratio_{B}GBps` | 12 |
| SO 前向 | 100、200 | `FSDP_fwd_so_comm_time_{B}GBps` | `FSDP_fwd_so_overlap_ratio_{B}GBps` | 4 |
| SO 反向 | 100、200 | `FSDP_bwd_so_comm_time_{B}GBps` | `FSDP_bwd_so_overlap_ratio_{B}GBps` | 4 |

以上 32 列按上表分项排列，每个分项内带宽升序，每档先耗时、后掩盖率。列名记录链路档位，现有 `*_bandwidth` 字段继续表示带宽需求。

随后追加以下 24 列，`{SU}`、`{SO}` 分别替换为对应带宽数值，单位均为 GB/s：

| 整体指标 | 字段名 | 列数 |
|---|---|---:|
| 前向 All-Gather | `FSDP_fwd_allgather_overlap_ratio_su{SU}GBps_so{SO}GBps` | 12 |
| 反向 Reduce-Scatter | `FSDP_bwd_reducescatter_overlap_ratio_su{SU}GBps_so{SO}GBps` | 12 |

按 SU 档位升序、SO 档位升序排列，每个组合先前向、后反向。例如 `FSDP_fwd_allgather_overlap_ratio_su400GBps_so100GBps`。全部 40 个掩盖率字段保存 0～1，显示为百分比。

## 校验

- k=1 时原有 SO 流量和带宽需求均为 0；新增 SO 通信耗时为 0，掩盖率为 100%。
- `带宽_GBps × 对应耗时_ms / 1000 = 流量_GB`。
- 同模型、精度、S、k 的结果相同，不再除以 EP、E-FSDP、E-DP。
- 独立按各通信阶段的分片传输量和精确层耗时复核；模型页与汇总页相同场景的字段一致。
- 基础 21 列与场景总表一致，两项层耗时与层耗时结果表一致。
- 核查 `通信耗时_ms × 链路带宽_GBps / 1000 = 对应流量_GB`，以及两种掩盖率公式一致。
- 每个分项的掩盖率随链路带宽单调不减且不超过 1；带宽达到对应需求时为 100%，零流量按上述约定处理。
- 整体掩盖率等于对应两个分项的最小值；独立以 `min(1, T_layer / max(V_SU/B_SU, V_SO/B_SO))` 复核，零流量按上述约定处理。
- 对照基线确认原有 31 列及场景数量不变，汇总页和各模型页同步追加 56 列。
