# E-FSDP：各 Rank 直接交互

状态：已实现。计算代码：[expert_fsdp_direct_exchange.py](../../../scripts/su_so_modeling/expert_fsdp_direct_exchange.py)。

## 输入与分组

沿用 [Expert-FSDP](expert_fsdp.md) 的参数、精度、SU 优先布局、域内/跨域组数和整层计算窗口：

```text
P_e = P_route/p
M_w = b_weight × P_e
M_g = b_gradient_comm × P_e
```

每张卡只与本 E-FSDP 组的其余 F−1 张卡交换分片；同 SU 用 SU，跨 SU 直接用 SO，不额外计入 SU 转发。前向向各成员发送自己的权重分片，反向向各目标 rank 发送其负责的梯度分片，由目标完成归约。跨域不采用域内复用或预归约。

## 流量与带宽

某个 SU 在该组中有 n_j 个成员，则其中每张卡的发送量与接收量分别相等：

```text
V_SU,j = (n_j−1)/F × M
V_SO,j = (F−n_j)/F × M
```

M 在前向取 M_w、反向取 M_g。不同 SU 的 n_j 可能不同，因此这些卡的 SU/SO 负载不一定均衡。场景记录对应链路的单卡最大值：

```text
c_SU = (n_max−1)/F
c_SO = (F−n_min)/F

V_fwd_SU = c_SU × M_w
V_fwd_SO = c_SO × M_w
V_bwd_SU = c_SU × M_g
V_bwd_SO = c_SO × M_g

BW_fwd_X = V_fwd_X / T_fwd
BW_bwd_X = V_bwd_X / T_bwd     # X 为 SU 或 SO
```

n_max、n_min 由相同的物理分组确定，SU 最大值与 SO 最大值可能来自不同卡。每卡的两链路发送量之和为 `(F−1)/F×M`，但两个场景最大值之和不一定满足该关系。

前向预取一层，反向采用稳态窗口；SU/SO 小块传输理想重叠，不计启动延迟、尾部和其他通信争用。

## 输出与掩盖率

独立输出 [results/E-FSDP_各Rank直接交互.xlsx](../../../results/E-FSDP_各Rank直接交互.xlsx)，共 92 列，包含基础场景、分组信息、四项流量、四项带宽、层耗时和分档掩盖率。统一字段及公式见[分档及输出规范](expert_fsdp_results.md)。

```bash
python3 scripts/generate_expert_fsdp_direct_exchange.py
```

使用 SU 的 400、600、800、1000、1200、1600 GB/s 和 SO 的 100、200 GB/s 档位，记录分项掩盖率及 12 个组合的前向/反向整体掩盖率。

## 校验

- 基础 21 列、分组信息和层耗时与另两种 E-FSDP 方式一致。
- 按实际卡的同域、跨域通信对象数量计数；发送量等于接收量，收发不相加。
- G_cross=0 时 SO 为 0；F=1 时两链路都为 0，零流量掩盖率为 100%。
- 按逐卡流量核查守恒，不将不同卡的 SU/SO 最大值当作同一卡的总量。
