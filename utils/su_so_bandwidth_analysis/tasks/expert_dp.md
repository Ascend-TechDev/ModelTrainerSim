# E-DP 专家副本同步

状态：已实现。分层计算：[expert_dp.py](../../../scripts/su_so_modeling/expert_dp.py)；物理分组：[expert_dp_layout.py](../../../scripts/su_so_modeling/expert_dp_layout.py)。三种通信方式分别输出独立结果，见[输出规范](expert_dp_results.md)。

## 输入与范围

沿用[场景输入](../scenario_inputs.md)、[模型输入](../model_inputs.md)和[整层计算耗时](layer_compute.md)。

| 符号 | 含义 |
|---|---|
| P_route | `10^9 × layer_routed_expert_total_params_b`，所选层全部路由专家参数 |
| p、F、R | EP、E-FSDP、E-DP 大小，`N=p×F×R=S×k` |
| q=S/p | 单个 SU 可容纳的完整 EP 组数量 |
| b_g | 梯度通信精度，FP8/BF16 训练均为 FP32，即 4 byte/参数 |
| T_bwd | 整层反向耗时，包含非专家和路由专家计算，为前向耗时的 2 倍 |

基线每次 micro-batch 反向对每层同步一次，不引入梯度累积间隔。同步顺序为：

```text
层专家梯度生成 → E-FSDP Reduce-Scatter → E-DP All-Reduce → optimizer
```

E-DP 在 R 个副本的相同专家分片间同步，更新后各副本持有相同的参数分片。本任务只计反向梯度 All-Reduce；不计共享专家、Router、Engram，也不追加前向通信。

```text
P_e = P_route/p
P_shard = P_route/(p×F)
M = b_g×P_shard = 4×P_route/(p×F)          # 单卡 All-Reduce 输入，byte
```

M 已经是 E-FSDP Reduce-Scatter 的输出大小，不再除以 F 或 R，也不按 Top-K 缩小。E-DP 的 All-Reduce 内部仍需归约、聚合两个阶段；它们与已经计入 E-FSDP 的 Reduce-Scatter 属于不同通信组。

## 物理分组

严格沿用 [EP → E-FSDP → E-DP 布局](expert_fsdp.md#物理分组先组成域内组再处理剩余成员)，不重新排列卡。固定逻辑坐标 f、e，变化副本编号 r，形成一个大小为 R 的 E-DP 组，共有 `p×F=N/R` 个组。

令 `h=floor(q/F)`、`u=q−h×F`。每个 SU 的 h 个完整本地副本分别为每个 E-DP 组贡献 h 个成员；跨域剩余副本按既定顺序再贡献成员。因为 `u<F`，一个 SU 对同一 E-DP 组最多再贡献 1 个成员。

- h≥1：每组覆盖全部 k 个 SU；其中 `k×u/F` 个 SU 有 h+1 个成员，其余 SU 有 h 个成员。
- h=0：每组覆盖 R 个 SU，每个 SU 只有 1 个成员。

因此所有 E-DP 组的成员数量分布相同，具体 SU 编号可以不同。记 d 为每组覆盖的 SU 数，m_min、m_max 只统计有成员的 SU：

```text
d = min(k,R)
m_min = max(1,floor(q/F))
m_max = ceil(q/F)

G_total = p×F = N/R
G_intra = G_total，若 d=1；否则为 0
G_cross = G_total−G_intra
```

上述公式依赖当前布局和 `F | q×k`，不适用于任意 rank 映射。当前布局下，一个场景的 E-DP 组全部域内或全部跨域；跨域组仍可能有多个 SU 内成员。R=1 的单成员组计入 G_intra，但无通信。

例：S=288、p=8、F=8、k=2，则 q=36、R=9。每个 E-DP 组在两个 SU 中的成员数为 4 和 5；不能套用 E-FSDP 的 4+4 跨域组，也不能把 9 个成员全部视为远端成员。

## 分层 All-Reduce

对一个 E-DP 组，设第 j 个 SU 有 m_j 个成员，`Σ_j m_j=R`。采用以下分解：

```text
SU Reduce-Scatter → SO Reduce-Scatter → SO All-Gather → SU All-Gather
```

SU 内先归约；SO 阶段将结果归约到 R 个最终分片，再按逆向路径分发。沿用理想小块分配，远端数据在目标 SU 的成员间均匀分摊。

| 阶段 | 每卡发送量 | 每卡接收量 |
|---|---|---|
| SU Reduce-Scatter | `(m_j−1)/m_j × M` | 同发送量 |
| SO Reduce-Scatter | `(1/m_j−1/R) × M` | `(d−1)/R × M` |
| SO All-Gather | `(d−1)/R × M` | `(1/m_j−1/R) × M` |
| SU All-Gather | `(m_j−1)/m_j × M` | 同发送量 |

完整 All-Reduce 的单卡单向流量为：

```text
V_SU,j = 2×(m_j−1)/m_j × M
V_SO,j = [1/m_j+(d−2)/R] × M
```

这里累加同一发送方向在两个阶段的流量，不将发送和接收相加。完整 All-Reduce 的发送总量与接收总量逐卡相等。成员不均匀时，SO 两阶段的收发负载会互换；应先分别累加发送、接收，再取最大值，不能将单阶段的 max(发送量,接收量) 直接乘 2。

场景结果分别取所有卡上的最大值：

```text
V_bwd_SU = 2×(m_max−1)/m_max × M
V_bwd_SO = [1/m_min+(d−2)/R] × M
```

SU 和 SO 的最大负载可能来自不同卡，二者不相加作为某一张卡的总流量。上述 4/5 成员例中，两项最大值分别为 `1.6×M`、`0.25×M`。

均匀分组时，`R=a×d`，每 SU 有 a 个成员，公式退化为：

```text
V_SU = 2×(a−1)/a × M
V_SO = 2×(d−1)/R × M
V_SU+V_SO = 2×(R−1)/R × M
```

最后一个等式仅适用于均匀组的逐卡流量；一般分组按全组总流量核查。All-Reduce 的均衡流量因子见 [NCCL Tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce)；SU/SO 分解由本节指定的布局和算法推导。

## 两种非分层方式

两种方式都在每个大小为 R 的 E-DP 组内执行，不在 N 卡上另组一个全局通信组；使用同一 M 和物理布局。

| 方式 | 单卡 SU 最大流量 | 单卡 SO 最大流量 |
|---|---|---|
| 组内全局 Ring | `2×(R−1)/R×M`，若 m_max>1；否则为 0 | `2×(R−1)/R×M`，若 d>1；否则为 0 |
| 各 Rank 直接交互 | `2×(m_max−1)/R×M` | `2×(R−m_min)/R×M` |

Ring 按同 SU 成员连续排列，Reduce-Scatter、All-Gather 均沿同一环方向执行。SO 指标为跨 SU 边界卡的最大单向负载，不摊到全部成员。

直接交互由一次分片 Reduce-Scatter 加一次分片 All-Gather 实现，每阶段向每个其他 rank 发送 M/R。域内用 SU，跨域直接用 SO，无额外 SU 转发；不是将完整 M 向每个 rank 广播。

## 掩盖窗口与带宽

沿用等层、整层反向稳态窗口：每层取当前最大 MoE 层，`τ=T_bwd`。E-DP 在该层 E-FSDP 归约后启动，可以与后续层计算重叠；不扣除启动等待，不计末层通信尾部。

```text
BW_bwd_SU = V_bwd_SU/T_bwd
BW_bwd_SO = V_bwd_SO/T_bwd
```

不把 E-FSDP 等待时间算作额外可用计算时间。当前分别估计各任务独立满足层间生产速率所需的带宽，不合并 E-FSDP、非专家 FSDP、EP 的链路争用；各项独立满足带宽并不代表联合通信一定完全掩盖。

SU/SO 沿用理想分块流水重叠。分档通信耗时表示稳态传输服务时间，不是包含各阶段依赖的单次 All-Reduce 延迟。SU 六档、SO 两档的分项及整体掩盖率见[输出规范](expert_dp_results.md)。

FP8/BF16 的梯度通信均为 FP32，因此相同模型与拓扑下，E-DP 流量相同；当前 FP8 层计算耗时为 BF16 的一半，所需带宽为其 2 倍。

## 输出与校验

三份独立结果表均沿用已有基础场景和 HBM 标记。M、V 输出 GB 时除以 10^9，层耗时及通信耗时输出 ms；字段见[输出规范](expert_dp_results.md)。

```bash
# 生成并核查三种 E-DP 方式
python3 scripts/generate_expert_dp.py
# 仅生成分层通信；--check 可只读核查已有结果
python3 scripts/generate_expert_dp.py --task expert_dp
```

- 按实际卡位置枚举 E-DP 组，组数为 p×F、每组 R 卡，每张卡恰好属于一个组；核查 d、m_min、m_max。
- 分层算法全组总发送量、总接收量均为 `2×(R−1)×M`；SU 总发送量为 `2×(R−d)×M`，SO 为 `2×(d−1)×M`。
- R=1 时全部流量为 0；k=1 时 SO 为 0；F≥q 时 SU 为 0；F=1 时同步完整 EP 专家单元。
- 同一布局下，SU 最大流量满足直接交互≤分层≤Ring；SO 满足分层≤直接交互≤Ring。
- 带宽、流量、窗口单位一致；掩盖率为 0～1，随链路带宽单调不减；零流量耗时为 0、掩盖率为 1。

独立校验从实际副本成员重新组成 E-DP 组，覆盖当前 5,196 种拓扑，验证布局闭式公式、流量守恒和上述边界；工作簿按精确分数参考核查全部字段及页签。执行结果见[核查记录](../validation.md)。
