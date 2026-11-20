# E-FSDP 专家参数通信

状态：已实现分层通信及分档掩盖率。计算代码：[expert_fsdp.py](../../../scripts/su_so_modeling/expert_fsdp.py)；物理布局：[expert_layout.py](../../../scripts/su_so_modeling/expert_layout.py)。

## 已确认范围

- 物理布局优先级为 **EP → E-FSDP → E-DP**：先保证 EP 组完整位于 SU 内，再优先在域内组成 E-FSDP，最后按完整 MoE 副本关系组成 E-DP。
- 仅统计前向权重 All-Gather、反向梯度 Reduce-Scatter；沿用此前的简化口径，不计反向权重 All-Gather。
- 本文定义分层通信：All-Gather 先 SO 后 SU，Reduce-Scatter 先 SU 后 SO；另设[组内全局 Ring](expert_fsdp_global_ring.md)和[各 Rank 直接交互](expert_fsdp_direct_exchange.md)，三种方式独立记录结果。
- 逐层掩盖，前向预取深度为 1；当前层与下一层均取输入中的最大 MoE 层。反向采用等层稳态窗口，不计通信尾部。
- 各通信任务单独估计需求，暂不合并与非专家 FSDP、EP、E-DP 的链路争用。
- 三种方式均按 SU 六档、SO 两档计算分项及整体掩盖率，统一使用[分档及输出规范](expert_fsdp_results.md)。

实际执行若前向后释放聚合权重，反向会再次收集权重，参见 [PyTorch FSDP 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。本轮将其排除是统计范围的简化，不据此假设完整权重一直驻留。

## 输入与参数范围

输入来自[模型字段](../model_inputs.md)、[基础场景](scenario_scan.md)、[精度及负载](../scenario_inputs.md)和[层耗时](layer_compute.md)：

| 符号 | 含义 |
|---|---|
| P_route | `10^9 × layer_routed_expert_total_params_b`，所选层全部路由专家参数 |
| E、K | 路由专家数、Top-K |
| p、F、R | EP、E-FSDP、E-DP 大小，`N=p×F×R=S×k` |
| q=S/p | 一个 SU 中完整 EP 组的数量 |
| b_weight | FP8/BF16 权重分别为 1/2 byte |
| b_gradient_comm | FP32 梯度通信，4 byte |
| T_fwd、T_bwd | 现有整层前向、反向计算时间，内部单位秒 |

```text
P_expert = P_route / E
P_e = (E/p) × P_expert = P_route/p
P_shard = P_e/F
M_w = b_weight × P_e
M_g = b_gradient_comm × P_e
```

每个 EP rank 负责 E/p 个专家，E-FSDP 在不同 EP 组的相同 EP rank 之间切分这组参数。All-Gather 的完整张量为 P_e，不按 Top-K 或 F 再次缩小。共享专家、Router 不属于本项。

R 个完整 MoE 副本各自执行 E-FSDP，R 不直接乘入或除入单卡通信量；副本间同步另计 [E-DP](expert_dp.md)。

## 物理分组：先组成域内组，再处理剩余成员

以一个完整 EP 组为布局单元，同一单元内 p 个 EP rank 始终留在同一 SU，按以下规则落实 SU 优先原则。

该顺序决定通信组成员的物理位置。每张卡同时属于一个 EP、E-FSDP 和 E-DP 组；`N=p×F×R` 只规定数量关系。一个 SU 可容纳多个完整 MoE 副本时，E-DP 也有域内成员；跨 SU 放置的副本之间使用 SO 同步。

1. 每个 SU 有 q 个 EP 组，先组成 `h=floor(q/F)` 个完整副本；每个副本含 F 个 EP 组，对应 p 个大小为 F 的 E-FSDP 组。
2. 每个 SU 剩余 `u=q−h×F` 个 EP 组。按 SU 编号、域内 EP 组编号排列剩余单元，每连续 F 个组成一个副本。
3. 副本编号为 r，副本内 EP 组编号为 f，EP rank 为 e。E-FSDP 固定 r、e，变化 f；E-DP 固定 f、e，变化 r，其物理布局由前两步确定。逻辑关系仍是 `N=p×F×R`。

由于 F 整除 q×k，剩余单元数 k×u 也整除 F，不丢弃卡或场景。F>q 时 h=0，全部 EP 组按第 2 步跨 SU 组建。

按实际 E-FSDP 通信组统计数量，每个完整 MoE 副本对应 p 个组：

```text
G_intra = p × k × floor(q/F)
G_cross = p × R − G_intra
G_total = G_intra + G_cross = p × R = N/F
```

G_cross=0 表示全部位于 SU 内；G_intra=0 表示全部跨 SU；两者均大于 0 表示混合分布。F=1 时有 N 个单成员组，通信量仍为 0。三种通信实现采用完全相同的物理分组和组数。

每个 E-FSDP 组 r 跨越 d_r 个 SU，在第 j 个 SU 内有 n_(r,j) 个成员：

```text
Σ_j n_(r,j) = F
1 ≤ n_(r,j) ≤ q
```

例：S=288、p=8、q=36、F=8、k=2。每个 SU 先组成 4 个域内副本，两个 SU 的剩余 4+4 个 EP 组再组成 1 个跨域副本，共 R=9 个副本。不能用一个统一的 `a=min(F,q)、b=F/a` 描述这些组。

## 分层通信实现与逐卡流量

对一个组省略 r，记其跨 d 个 SU，第 j 个 SU 有 n_j 个成员。M 表示完整的专家权重 M_w 或梯度 M_g。

前向 SO 阶段，每张卡将自己原有的 M/F 权重分片向其余 d−1 个 SU 各发送一份。发往某个 SU 的数据按小块均匀分配给该 SU 的 n_j 个成员；每份数据仅跨域一次，不向远端每张卡重复发送。随后在 SU 内 All-Gather。

反向先在各 SU 的 n_j 个成员内 Reduce-Scatter，再将域内归约结果的分片发给最终负责它的 rank，由该 rank 汇总来自其他 SU 的结果。它是上述聚合过程的逆向分配。

在理想分块、无填充的假设下，第 j 个 SU 中每张卡的通信量为：

| 通信阶段 | 单卡发送量 | 单卡接收量 |
|---|---|---|
| 前向 SO All-Gather | `(d−1)/F × M_w` | `(1/n_j−1/F) × M_w` |
| 前向 SU All-Gather | `(n_j−1)/n_j × M_w` | 同发送量 |
| 反向 SU Reduce-Scatter | `(n_j−1)/n_j × M_g` | 同发送量 |
| 反向 SO Reduce-Scatter | `(1/n_j−1/F) × M_g` | `(d−1)/F × M_g` |

说明：前向 SO 阶段后，每张卡持有 `M_w/F+(1/n_j−1/F)×M_w=M_w/n_j`，可继续执行域内聚合。反向域内归约后每张卡持有 M_g/n_j，最终保留 M_g/F，其余部分跨域发送。

不均匀分组时，同一卡的 SO 发送量和接收量可能不同。带宽需求按**单卡单向最大负载**计算，即取发送、接收两者中的较大者，再取整个场景所有卡的最大值；收发不相加。

## 场景最大流量与带宽

设 n_max、n_min 是本场景所有 E-FSDP 组中非空 SU 成员数量的最大、最小值。按上一节的分组规则，可得：

```text
n_max = min(F,q)
n_min = gcd(F,q)

c_SU = (n_max−1)/n_max
c_SO = 1/n_min−1/F
```

依据：剩余单元按等长区间切分，组边界与 SU 剩余单元边界的最小交叠为 gcd(F,q)；F 整除 q×k 保证完整覆盖其边界周期。实现时仍需枚举实际组成员核查上述极值。

SO 中，最少成员 SU 的前向接收量、反向发送量达到 c_SO×M；其他 SU 的发送/接收量不会超过该值。SU 和 SO 的最大负载可能来自不同卡，不能将两个最大值相加当作某一张卡的总流量。

| 指标 | 单卡单向最大流量 | 带宽需求 |
|---|---|---|
| 前向 SU All-Gather | `V_fwd_SU=c_SU×M_w` | `V_fwd_SU/T_fwd` |
| 前向 SO All-Gather | `V_fwd_SO=c_SO×M_w` | `V_fwd_SO/T_fwd` |
| 反向 SU Reduce-Scatter | `V_bwd_SU=c_SU×M_g` | `V_bwd_SU/T_bwd` |
| 反向 SO Reduce-Scatter | `V_bwd_SO=c_SO×M_g` | `V_bwd_SO/T_bwd` |

M、V 内部单位为 byte，输出 GB、GB/s 时除以 10^9；已有层耗时的 ms 需除以 1000 后代入。带宽为单卡有效带宽，SU/SO 小块传输理想重叠，不额外折减网络利用率或加入启动延迟。

若某组均匀分布在 b 个 SU、每个 SU 有 a 个成员，则 F=a×b，退化为熟悉的分层公式：

```text
V_SU = (a−1)/a × M
V_SO = (b−1)/(a×b) × M
V_SU+V_SO = (F−1)/F × M
```

最后一个等式仅是均匀组的逐卡关系。非均匀布局应检查组内全部卡的总发送量为 `(F−1)×M`，而非要求两个场景最大值之和满足逐卡均衡公式。

## 计算量与掩盖窗口

均衡路由时，一个 EP 组有 p×t 个原始 token，每个专家处理 `p×t×K/E` 次分配，每卡计算 E/p 个专家：

```text
C_moe_fwd = 2 × P_expert × (p×t×K/E) × (E/p)
          = 2 × P_expert × t × K
```

E-FSDP 切分存储，不切分单个专家的矩阵计算；聚合后各 rank 处理各自 EP 组的 token，计算量不再除以 p、F 或 R。

沿用逐层掩盖，窗口使用已有的整层 T_fwd、T_bwd，包含非专家及路由专家计算；不因为只统计专家通信而缩小为专家计算时间。反向窗口仍为前向的 2 倍，暂不计额外重计算和 EP 层内掩盖。

## 输出与校验

分层通信独立输出 [results/E-FSDP.xlsx](../../../results/E-FSDP.xlsx)，共 92 列；同一场景仍占一行，记录四项单卡最大带宽、域内/跨域组数、分档耗时和掩盖率。全局 Ring、直接交互各自使用独立文件，不合并到分层表。完整字段、带宽档位、掩盖率公式及文件名见[分档及输出规范](expert_fsdp_results.md)。

```bash
# 生成并核查三种方式
python3 scripts/generate_expert_fsdp.py
# 仅生成分层通信
python3 scripts/generate_expert_fsdp.py --task expert_fsdp
```

校验包括：

- 参数与组规模：`P_e×p=P_route`、每个组的成员数为 F、所有 rank 恰好属于一个相应组。
- 实际通信组数量为 p×R，域内组和跨域组数量之和为 N/F；不能将完整 MoE 副本数 R 当作 E-FSDP 通信组数。
- 按实际物理布局枚举 n_(r,j)，核查 n_min、n_max；检查不规则整除场景及收发两侧。
- 组内总收发守恒：每次 AG/RS 的总发送量和总接收量均为 `(F−1)×M`。
- F=1 时 SU/SO 均为 0；F 整除 q 时全部域内、SO 为 0；q 整除 F 时退化为每 SU 有 q 个成员的均匀分层。
- 精度、窗口和单位一致；基础场景及层耗时与已有结果表一致。
- 分项及整体掩盖率按[统一校验规则](expert_fsdp_results.md#校验)复核。

当前 HBM 标记只覆盖基础分片训练状态，不包含聚合权重、通信缓冲区等瞬时峰值；本轮不更改该容量判定口径。
