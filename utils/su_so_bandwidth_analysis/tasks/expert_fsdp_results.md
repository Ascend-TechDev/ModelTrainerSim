# E-FSDP 分档掩盖率与输出规范

状态：已实现；三种通信方式使用相同的 92 列输出契约。

适用于[分层通信](expert_fsdp.md)、[组内全局 Ring](expert_fsdp_global_ring.md)、[各 Rank 直接交互](expert_fsdp_direct_exchange.md)。三种实现使用相同的基础场景、物理分组、权重/梯度精度和整层掩盖窗口，仅通信量计算方式不同。

## 场景记录口径

每个基础扫描场景仍占一行，按整个场景所有卡的单向最大负载记录四项流量和带宽。发送与接收取最大值，不相加；不按卡数或组数平均。SU 和 SO 的最大值可能位于不同组、不同卡。

| 分布 | SU 字段 | SO 字段 |
|---|---|---|
| 全部域内组 | 域内组中单卡最大值 | 0 |
| 域内组、跨域组并存 | 两类组中单卡最大值 | 跨域组中单卡最大值 |
| 全部跨域组 | 跨域组中单卡最大值 | 跨域组中单卡最大值 |

记录实际通信组数量：`G_intra=p×k×floor(q/F)`、`G_cross=p×R−G_intra`，两者之和为 `p×R=N/F`。组数不额外乘入单卡最大流量或带宽。

## 链路档位、耗时与掩盖率

| 链路 | 单卡单向可用带宽（GB/s） |
|---|---|
| SU | 400、600、800、1000、1200、1600 |
| SO | 100、200 |

带宽直接作为有效带宽使用，不额外折减网络利用率。档位按列记录，不扩展场景行数。每个档位分别计算前向和反向：

```text
T_comm_fwd_X = V_fwd_X / B_X
T_comm_bwd_X = V_bwd_X / B_X
rho_fwd_X = min(1, T_fwd / T_comm_fwd_X)
rho_bwd_X = min(1, T_bwd / T_comm_bwd_X)
```

X 为 SU 或 SO；V 使用该方式的单卡最大流量，单位 GB，B 单位 GB/s，时间为秒。反向分母使用反向通信耗时。V=0 时，通信耗时为 0、掩盖率约定为 1。

前向按层间预取、反向按稳态流水，SU/SO 理想重叠。12 个 SU/SO 组合分别记录整体掩盖率：

```text
rho_AllGather(B_SU,B_SO) = min(rho_fwd_SU, rho_fwd_SO)
rho_ReduceScatter(B_SU,B_SO) = min(rho_bwd_SU, rho_bwd_SO)
```

等价地，取 `T_comm_total=max(V_SU/B_SU,V_SO/B_SO)`，再计算 `min(1,T_layer/T_comm_total)`；两链路均为零时结果为 1。虽然两个最大负载可能来自不同卡，整体指标仍反映最忙链路/卡对场景通信完成时间的限制。它是当前理想假设下的最差掩盖率，不是卡或组的平均掩盖率。

通信耗时输出 ms；掩盖率保存为 0～1、显示百分比。全部场景保留，使用时按 HBM 合法标记筛选；不计其他通信任务争用。

## 独立结果文件

三种方式分别输出，均包含全部场景：

| 通信方式 | 输出文件 | 列数 |
|---|---|---:|
| 分层通信 | [results/E-FSDP.xlsx](../../../results/E-FSDP.xlsx) | 92 |
| 组内全局 Ring | [results/E-FSDP_全局Ring.xlsx](../../../results/E-FSDP_全局Ring.xlsx) | 92 |
| 各 Rank 直接交互 | [results/E-FSDP_各Rank直接交互.xlsx](../../../results/E-FSDP_各Rank直接交互.xlsx) | 92 |

每份表第一张为“全部场景”，之后每个模型一个页签。三个文件共享场景 ID、行序、基础 21 列、5 项分组信息及两项层耗时，通信实现记录在文件名、说明和工作簿属性中。不会向已有非专家 FSDP 结果表追加字段。

`python3 scripts/generate_expert_fsdp.py` 独立生成并校验三表；加 `--check` 只核查已有结果。统一入口也支持 `--task expert_fsdp_all`，或选择 `expert_fsdp`、`expert_fsdp_global_ring`、`expert_fsdp_direct_exchange` 单独执行。

## 字段及顺序

| 字段组 | 列数 |
|---|---:|
| 基础场景及 HBM 检查 | 21 |
| q、n_min、n_max | 3 |
| 域内/跨域 E-FSDP 组数量 | 2 |
| 前向/反向层耗时 | 2 |
| 前向/反向 SU/SO 流量及带宽需求 | 8 |
| 分档通信耗时 | 16 |
| 分档分项掩盖率 | 16 |
| 12 个组合的整体掩盖率 | 24 |
| 合计 | 92 |

基础 21 列之后，依次记录以下 15 列：

| 字段 | 含义 / 单位 |
|---|---|
| `EFSDP_ep_groups_per_su` | q=S/p，个 |
| `EFSDP_su_members_min` | n_min=gcd(F,q)，组内单 SU 成员数最小值，个 |
| `EFSDP_su_members_max` | n_max=min(F,q)，组内单 SU 成员数最大值，个 |
| `EFSDP_intra_su_group_count` | G_intra，完全位于单个 SU 的实际通信组数量，个 |
| `EFSDP_cross_su_group_count` | G_cross，跨 SU 的实际通信组数量，个 |
| `layer_fwd_time` | 整层前向掩盖窗口，ms |
| `layer_bwd_time` | 整层反向稳态窗口，ms |
| `EFSDP_fwd_su_vol` | 前向 SU 单卡单向最大流量，GB |
| `EFSDP_fwd_so_vol` | 前向 SO 单卡单向最大流量，GB |
| `EFSDP_fwd_su_bandwidth` | 前向 SU 带宽需求，GB/s |
| `EFSDP_fwd_so_bandwidth` | 前向 SO 带宽需求，GB/s |
| `EFSDP_bwd_su_vol` | 反向 SU 单卡单向最大流量，GB |
| `EFSDP_bwd_so_vol` | 反向 SO 单卡单向最大流量，GB |
| `EFSDP_bwd_su_bandwidth` | 反向 SU 带宽需求，GB/s |
| `EFSDP_bwd_so_bandwidth` | 反向 SO 带宽需求，GB/s |

随后追加 32 列分档分项结果：

| 分项 | B 档位（GB/s） | 通信耗时字段（ms） | 掩盖率字段（0～1） | 列数 |
|---|---|---|---|---:|
| SU 前向 | 400、600、800、1000、1200、1600 | `EFSDP_fwd_su_comm_time_{B}GBps` | `EFSDP_fwd_su_overlap_ratio_{B}GBps` | 12 |
| SU 反向 | 400、600、800、1000、1200、1600 | `EFSDP_bwd_su_comm_time_{B}GBps` | `EFSDP_bwd_su_overlap_ratio_{B}GBps` | 12 |
| SO 前向 | 100、200 | `EFSDP_fwd_so_comm_time_{B}GBps` | `EFSDP_fwd_so_overlap_ratio_{B}GBps` | 4 |
| SO 反向 | 100、200 | `EFSDP_bwd_so_comm_time_{B}GBps` | `EFSDP_bwd_so_overlap_ratio_{B}GBps` | 4 |

按上表分项顺序、每项内 B 升序排列，每档先耗时、后掩盖率。例如 `EFSDP_fwd_su_overlap_ratio_400GBps`。

最后追加 24 列整体掩盖率：

| 字段模板 | 列数 |
|---|---:|
| `EFSDP_fwd_allgather_overlap_ratio_su{SU}GBps_so{SO}GBps` | 12 |
| `EFSDP_bwd_reducescatter_overlap_ratio_su{SU}GBps_so{SO}GBps` | 12 |

SU 档位升序、SO 档位升序，每个组合先前向、后反向。每份表共有 40 项掩盖率：16 项分项、24 项整体。

## 校验

- 三表基础、布局、层耗时及行序一致；汇总页与模型页一致。
- 按各自逐卡流量模型独立复核场景最大值，域内/跨域组数之和为 N/F。
- `带宽需求_GBps × 层耗时_ms/1000 = 流量_GB`；`档位带宽_GBps × 通信耗时_ms/1000 = 流量_GB`。
- 掩盖率为 0～1，随对应链路带宽单调不减；带宽达到需求时为 1；零流量耗时为 0、掩盖率为 1。
- 整体掩盖率等于两链路分项的最小值，并以两链路通信耗时最大值独立复核。
- 同布局下，SU 最大流量满足直接交互≤分层≤Ring；SO 最大流量满足分层≤直接交互≤Ring。
- F=1 时全部通信为零；无跨域组时 SO 为零；非均匀布局需验证收发两侧及不同卡上的最大值。
