# E-FSDP：组内全局 Ring

状态：已实现。计算代码：[expert_fsdp_global_ring.py](../../../scripts/su_so_modeling/expert_fsdp_global_ring.py)。

## 输入与分组

沿用 [Expert-FSDP](expert_fsdp.md) 的参数、精度、SU 优先布局、域内/跨域组数和整层计算窗口。每个组聚合该 EP rank 负责的全部专家参数：

```text
P_e = P_route/p
M_w = b_weight × P_e
M_g = b_gradient_comm × P_e
```

每个 E-FSDP 组的 F 张卡单独组成一个 Ring，同一 SU 的成员连续排列。“全局”指覆盖该组的全部 F 张卡，不是全部 N 张卡；不同 EP rank、不同 E-DP 副本之间不串成一个环。

## 流量与带宽

每条环边发送 F−1 个大小为 M/F 的分片，M 在前向取 M_w、反向取 M_g。SO 仅发生在跨 SU 环边的出口发送和入口接收，内部非边界卡没有 SO 流量。

按各链路的单卡最大发送/接收负载记录，收发不相加：

```text
c_SU = (F−1)/F，若 n_max > 1；否则为 0
c_SO = (F−1)/F，若 G_cross > 0；否则为 0

V_fwd_SU = c_SU × M_w
V_fwd_SO = c_SO × M_w
V_bwd_SU = c_SU × M_g
V_bwd_SO = c_SO × M_g

BW_fwd_X = V_fwd_X / T_fwd
BW_bwd_X = V_bwd_X / T_bwd     # X 为 SU 或 SO
```

n_max 为本场景所有组中，单个 SU 内成员数的最大值。只要存在 SU 内相邻成员，场景 SU 最大值就达到该环边负载；只要有跨域组，SO 最大值就取其边界卡负载。二者可以来自不同组、不同卡。

前向预取一层，反向采用稳态窗口；SU/SO 小块传输理想重叠，不计环步进延迟、多 Ring 出口均衡、启动和尾部。此模型不以场景平均流量代替边界卡负载。

## 输出与掩盖率

独立输出 [results/E-FSDP_全局Ring.xlsx](../../../results/E-FSDP_全局Ring.xlsx)，共 92 列，包含基础场景、分组信息、四项流量、四项带宽、层耗时和分档掩盖率。统一字段及公式见[分档及输出规范](expert_fsdp_results.md)。

```bash
python3 scripts/generate_expert_fsdp_global_ring.py
```

使用 SU 的 400、600、800、1000、1200、1600 GB/s 和 SO 的 100、200 GB/s 档位，记录分项掩盖率及 12 个组合的前向/反向整体掩盖率。

## 校验

- 基础 21 列、分组信息和层耗时与另两种 E-FSDP 方式一致。
- 每个实际组按其环边分类验证 SU/SO 最大值；总发送量为 `(F−1)×M`。
- G_cross=0 时 SO 流量、带宽和耗时为 0，SO 掩盖率为 100%；F=1 时两链路都无通信。
- 域内组和跨域组并存时，分别取各链路最忙卡，不要求两个场景最大值来自同一卡。
