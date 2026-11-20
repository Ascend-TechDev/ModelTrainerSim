# E-DP 分档掩盖率与输出规范

状态：已实现；三种通信方式使用相同的 62 列输出契约。计算方法见 [E-DP](expert_dp.md)。

## 独立结果表

| 通信方式 | 输出文件 | 列数 |
|---|---|---:|
| 分层通信 | [results/E-DP.xlsx](../../../results/E-DP.xlsx) | 62 |
| 组内全局 Ring | [results/E-DP_全局Ring.xlsx](../../../results/E-DP_全局Ring.xlsx) | 62 |
| 各 Rank 直接交互 | [results/E-DP_各Rank直接交互.xlsx](../../../results/E-DP_各Rank直接交互.xlsx) | 62 |

每份表保留基础 21 列、相同场景 ID 及行序；第一页为全部场景，其后每模型一页。全部场景保留，分析时按 HBM 合法标记筛选。三表布局和层耗时相同，各自使用对应算法的流量；不向已有任务表追加字段。

`python3 scripts/generate_expert_dp.py` 独立生成并校验三表；加 `--check` 只读核查。统一入口支持 `--task expert_dp_all`，或选择 `expert_dp`、`expert_dp_global_ring`、`expert_dp_direct_exchange` 单独执行。

## 分档耗时与掩盖率

| 链路 | 单卡单向有效带宽（GB/s） |
|---|---|
| SU | 400、600、800、1000、1200、1600 |
| SO | 100、200 |

只计算反向 All-Reduce。V 使用单卡单向最大流量，单位 GB；B 使用 GB/s，以下时间为秒：

```text
T_comm_bwd_SU = V_bwd_SU/B_SU
T_comm_bwd_SO = V_bwd_SO/B_SO
rho_bwd_SU = min(1,T_bwd/T_comm_bwd_SU)
rho_bwd_SO = min(1,T_bwd/T_comm_bwd_SO)

rho_AllReduce(B_SU,B_SO) = min(rho_bwd_SU,rho_bwd_SO)
```

整体指标基于 SU/SO 理想流水重叠，等价于使用两条链路服务时间的最大值；不将两者相加。V=0 时该链路耗时为 0、掩盖率为 1。档位按列展开，场景行数不变。

## 字段及顺序

基础 21 列后依次增加以下 13 列：

| 字段 | 含义 / 单位 |
|---|---|
| `EDP_ep_groups_per_su` | q=S/p，个 |
| `EDP_su_domain_count` | 每个 E-DP 组覆盖的 SU 数 d，个 |
| `EDP_su_members_min` | 非空 SU 中的最少组成员数 m_min，个 |
| `EDP_su_members_max` | 非空 SU 中的最多组成员数 m_max，个 |
| `EDP_intra_su_group_count` | 完全域内的 E-DP 组数量，个 |
| `EDP_cross_su_group_count` | 跨 SU 的 E-DP 组数量，个 |
| `layer_fwd_time` | 沿用整层前向耗时，ms |
| `layer_bwd_time` | 整层反向稳态窗口，ms |
| `EDP_gradient_shard_vol` | 单卡 All-Reduce 输入 M，GB |
| `EDP_bwd_su_vol` | SU 单卡单向最大流量，GB |
| `EDP_bwd_so_vol` | SO 单卡单向最大流量，GB |
| `EDP_bwd_su_bandwidth` | SU 带宽需求，GB/s |
| `EDP_bwd_so_bandwidth` | SO 带宽需求，GB/s |

随后按 SU、SO 顺序和档位升序，每档先耗时、后掩盖率，追加 16 列：

| 链路 | 通信耗时字段（ms） | 分项掩盖率字段（0～1） | 列数 |
|---|---|---|---:|
| SU 六档 | `EDP_bwd_su_comm_time_{B}GBps` | `EDP_bwd_su_overlap_ratio_{B}GBps` | 12 |
| SO 两档 | `EDP_bwd_so_comm_time_{B}GBps` | `EDP_bwd_so_overlap_ratio_{B}GBps` | 4 |

最后按 SU 档位升序、SO 档位升序，追加 12 列：

```text
EDP_bwd_allreduce_overlap_ratio_su{SU}GBps_so{SO}GBps
```

共 `21+13+16+12=62` 列，其中 8 项通信耗时、8 项分项掩盖率和 12 项整体掩盖率。掩盖率保存为 0～1，显示百分比；本任务无前向通信字段。

## 校验

- 基础 21 列、场景行序、两项层耗时与已有结果一致；汇总页和模型页数据一致。
- `带宽需求×层反向耗时_ms/1000=流量`，`档位带宽×通信耗时_ms/1000=流量`。
- 域内/跨域组数之和为 p×F，组数不乘入单卡通信量。
- 分项掩盖率在带宽达到需求时为 1，整体掩盖率等于两链路分项的最小值。
- 核查不均匀组完整 All-Reduce 的逐卡流量，再取场景最大值；不能用两倍单阶段收发最大值代替完整操作流量。
