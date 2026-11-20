# Engram 按需查表通信

按行分片、均衡访问，仅分析**参数量最大的 MoE 层所对应的一张 Engram 表**，独立输出 [Engram.xlsx](../../../results/Engram.xlsx)。字段见[输出规范](engram_results.md)。

## 输入与范围

| 符号 | 来源 / 含义 |
|---|---|
| l | `layer_id_zero_based`，所选最大 MoE 层号 |
| L | `layer_engram_table_rows`，所选表的记录数 |
| D | `layer_engram_record_params`，每条记录的参数数 |
| P | `10^9 × layer_engram_table_params_b`，所选表参数数，`P=L×D` |
| S、k、N | SU 卡数、SU 数、总卡数，`N=S×k` |
| t | 单卡 micro-batch×seq_len，2048 token |
| λ | 每 token 对所选表的查询次数，24 |
| b_w | FP8：1；BF16：2 byte/参数 |
| b_g、b_id | FP32 梯度：4 byte/参数；行索引：8 byte |
| T_fwd、T_bwd | 已有整层计算耗时，反向为前向的 2 倍，公式见[层计算](layer_compute.md) |

当前五个有 Engram 的模型为 `dsv41flash`、`dsv41next10t`、`dsv41next20t`、`dsv41next40t`、`dsv41next80t`；所选表均位于零起始第 14 层，D=256。模型中的其他 Engram 表不参与本任务通信计算，也不合并共享链路需求。

CSV/JSON 不含 λ 和索引字节数，本任务显式声明 λ=24、b_id=8。已核对报告中的 `details.hashes_per_token` 和 `hash_row_ids.element_bytes`；[仓库实现](../../../ModelTrainerSim/src/models/dsv41flash/modules/engram.py)给出 `λ=(engram_max_ngram_size−1)×engram_n_heads=(4−1)×8`。代码使用该声明，不在运行时从报告补字段。

Engram 投影/门控已计入非专家 FSDP 和层计算量；查表权重不追加矩阵计算 FLOPs。HBM 检查仍包含**全模型所有表**的完整训练状态，沿用原有判定，不因只统计一张表的通信而缩减容量。

## 分片与访问

- 表按行均匀分布到 N 卡，每条记录由唯一 owner 持有；不再按 EP、E-FSDP、E-DP 切分或复制。
- 无缓存、无请求去重、无传输前梯度合并；重复命中按查询次数计流量。Q/N 表示均衡访问的平均请求数，可为非整数。
- 同 SU 的其他卡走 SU，其他 SU 的卡直接走 SO，不追加 SU 中转流量。
- 只计有效载荷；忽略包头、量化元数据、启动/往返延迟、随机查表耗时及其他任务争用。

训练时将表分片到 GPU、通过 All-to-All 获取记录并回传梯度，参见 [Engram 论文 §2.5](https://arxiv.org/html/2601.07372v1#S2.SS5)。下列均衡访问及载荷协议为本分析的假设。

```text
Q = t×λ = 2048×24 = 49,152         # 每卡查询次数
p_local = 1/N                      # 本卡命中，不通信
p_SU = (S−1)/N
p_SO = (N−S)/N
p_local+p_SU+p_SO = 1
```

Q 不乘 MoE Top-K 或并行组大小。D、λ 相同且无缓存时，整表记录数不影响单次 micro-batch 的查询载荷。

## 单卡前向流量

请求方发送 b_id byte 行索引，owner 返回 D×b_w byte 记录；返回顺序与请求顺序一致，无额外返回索引。每卡既发起查询，也向其他卡提供记录。均衡条件下发送量与接收量相等，按**单向流量**记录：

```text
V_fwd_SU = Q×(S−1)/N×(b_id+D×b_w)
V_fwd_SO = Q×(N−S)/N×(b_id+D×b_w)
```

索引和记录两类载荷相加，发送与接收不相加，不乘 2。记录使用本次训练的权重精度。

## 单卡反向流量

请求方发送 b_id byte 索引和 D×b_g byte FP32 梯度，owner 按行汇总。不假设 owner 保留前向索引映射，因此仍计索引；不追加响应或整表梯度 Collective：

```text
V_bwd_SU = Q×(S−1)/N×(b_id+D×b_g)
V_bwd_SO = Q×(N−S)/N×(b_id+D×b_g)
```

优化器内部通信不在本任务范围。上述 V 单位为 byte，输出转换为 GB=10^9 byte。

## 前向预取与反向窗口

查询索引只依赖输入 token 和确定性哈希，可以提前确定；查表记录跨层预取，投影和门控仍在对应层执行。[论文 §2.5](https://arxiv.org/html/2601.07372v1#S2.SS5)

各层计算时长均近似为已有最大 MoE 层的时长，采用用户指定的完整窗口：

| 方向 | 窗口层数 d | 可掩盖计算窗口 |
|---|---|---|
| 前向 | 1、5、10、15、20 | W_fwd(d)=d×T_fwd |
| 反向 | 1、5、10、15、20 | W_bwd(d)=d×T_bwd |

前向假设本次所需版本的表权重已可读取，前向开始前也有足够计算时间提供完整预取窗口；**不按目标层号截断窗口**。反向假设梯度产生后、optimizer 前有对应的完整回传预算；索引已知不代表梯度可提前产生。本阶段不校验真实调度、边界层预算或通信尾部。

只考虑所选一张表的通信，不累加多张表的流量，也不建模多表共享链路。延长窗口不改变流量：

```text
BW_fwd_X(d) = V_fwd_X / (d×T_fwd)
BW_bwd_X(d) = V_bwd_X / (d×T_bwd)       # X∈{SU,SO}
```

V 用 GB、T 用秒时，BW 为 GB/s；十组窗口共输出 20 项 SU/SO 带宽需求。前向1层的带宽需求为前向5层的5倍，流量和通信耗时不变。

## 分档掩盖率

SU 档位为 400、600、800、1000、1200、1600 GB/s；SO 为 100、200 GB/s，均为单卡单向有效带宽。

```text
T_comm_phase_X = V_phase_X / B_X
rho_phase_X(d) = min(1, W_phase(d) / T_comm_phase_X)
rho_phase_total(d,B_SU,B_SO) = min(rho_phase_SU(d),rho_phase_SO(d))
```

通信耗时不随窗口层数改变。整体掩盖率沿用 SU/SO 理想流水重叠。零流量时耗时为 0、掩盖率为 1；无对应 Engram 表时结构、查询数、窗口、流量和带宽均填 0，共有层耗时仍保留。

## 执行与校验

```bash
python3 scripts/generate_engram.py
python3 scripts/generate_engram.py --check
```

输入解析、计算和独立核查分别位于 [engram_inputs.py](../../../scripts/su_so_modeling/engram_inputs.py)、[engram.py](../../../scripts/su_so_modeling/engram.py)、[engram_validation.py](../../../scripts/su_so_modeling/engram_validation.py)。支持根目录 CSV/JSON，不修改模型输入。

- 核查所选表 P=L×D，并与模型级同一层的表字段一致；不取多表之和。
- 逐卡请求、响应、梯度的发送和接收守恒；k=1 时 SO 为 0，N=1 时网络流量为 0。
- 全部拓扑独立复核：V_SU+V_SO=Q×(N−1)/N×(b_id+D×b_phase)。
- 固定模型、精度、S/k 时，EP、E-FSDP、E-DP 不改变本任务流量或窗口。
- 带宽×窗口=流量；带宽随窗口层数成反比，掩盖率随窗口和链路带宽非递减。
- 两种精度反向流量相同，同深度下 FP8 反向需求为 BF16 的 2 倍；前向比为 `2×(8+D)/(8+2D)`，D=256 时约 1.01538。
- 全部页签核查基础 21 列、行序、两项层耗时及汇总/模型页数据一致性。
