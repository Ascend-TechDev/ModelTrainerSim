# EP All-to-All：前向层内掩盖

独立输出 [EP.xlsx](../../../results/EP.xlsx)。仅分析前向 dispatch/combine；EP 完整位于 SU 内，SO 流量和带宽均为 0。采用中间维、EP Size、精度组合的维度扫描，不展开全量集群场景。

## 输入与扫描因子

| 因子 | 来源 / 当前取值 |
|---|---|
| MoE 中间维 m | 根目录 [CSV](../../../max_layer_parameters.csv) 或 [JSON](../../../max_layer_parameters.json) 的 `moe_expert_hidden_size` 去重、升序：2048、2304、3072、4096、4608、6144、8192 |
| EP Size p | [公共输入](../scenario_inputs.md)的 EP 候选值：8、16、32、64、128、256、512、1024、2048 |
| EP 精度组合 | FP8、BF16、DeepEP，分别保留 |

全量笛卡尔积为 **7×9×3=189** 个场景，每种精度 63 个；六档 SU 带宽按列展开。FP8/BF16 结果相同，但不合并场景。

本表不按某个来源模型的专家数筛选，也不做 HBM 检查。应用到具体模型时，仍须满足 p 整除专家数 E、p<SU Size、p 整除 SU Size，以及原容量约束。中间维来源页记录模型及专家数，供追溯。

## 固定配置与三种精度组合

峰值算力从公共输入读取；本任务的可执行配置如下：

<!-- ep-scan-config -->
```json
{
  "gemm_utilization": 0.95,
  "all2all_bandwidth_utilization": 0.70,
  "su_bandwidths_gbps": [400, 600, 800, 1000, 1200, 1600]
}
```

| EP 场景 | Dispatch 字节数 b_D | Combine 字节数 b_C | GEMM 精度 | 峰值算力 | 95% 有效算力 F |
|---|---:|---:|---|---:|---:|
| FP8 | 1（FP8） | 1（FP8） | FP8 | 3538 TOPS | 3361.1 TOPS |
| BF16 | 2（BF16） | 2（BF16） | BF16 | 1769 TOPS | 1680.55 TOPS |
| DeepEP | 1（FP8） | 2（BF16） | FP8 | 3538 TOPS | 3361.1 TOPS |

95% 是 GEMM 利用率，70% 是 All2All 的链路带宽利用率，分别作用于计算时间和通信时间。已有整层计算及其他任务仍沿用原来的 40% 计算口径。

DeepEP 仅代表用户指定的精度组合，不复现库内布局、去重、调度或 SM 占用。[官方说明](https://github.com/deepseek-ai/DeepEP#example-use-in-model-training-and-inference)支持 FP8 dispatch，combine 输入为 BF16。三个场景均忽略量化尺度、索引、路由权重等元数据和量化/反量化开销。

## 层内计算窗口

当前路由专家为三个无 bias 矩阵：gate/up 各为 [d,m]，proj 为 [m,d]，一个专家参数量为 `P_e=3dm`。gate 指专家 FFN 的门控投影，不是 MoE Router；proj 是专家输出投影，也称 down projection。gate/up 即使融合调用，仍计两个 GEMM。

普通 MoE 的 d 等于 hidden size；Kimi 在 Latent 降维后 dispatch、combine 后升维，d 是 latent 维。它们使用同一推导，d 最终抵消，不需要作为结果场景因子。当前 16 个模型的三矩阵结构已经核对；结构改变时需重新验证。

仅使用 gate+up 的 GEMM 时间掩盖 dispatch，proj 的 GEMM 时间掩盖 combine；不借用整层或其他层计算。掩盖发生在同层不同 token 块之间：

| 计算窗口 | 同时进行的通信 |
|---|---|
| gate+up(i) | dispatch(i+1) |
| proj(i) | combine(i−1)，其输出已就绪 |

假设支持分块启动、缓冲充足、均衡路由和 95% GEMM 利用率；忽略流水启动/排空、同步延迟及其他任务争用。同一块仍需遵守 dispatch → GEMM → combine 的依赖。

不计 token 丢弃、padding、同目的 rank 的去重和结果预合并；同一 token 命中同 rank 的多个专家，仍按多次 token→expert 分配计载荷。Router、共享专家、Engram、Latent 投影、激活函数和逐元素运算均不计入这两个 GEMM 窗口。

## 流量与计算量推导

令每卡原始 token 数为 t、Top-K 为 K、专家数为 E。完全均衡时，每卡处理的分配数为：

```text
n = (E/p)×(p×t×K/E) = t×K
α = (p−1)/p

V_D_SU = α×n×d×b_D
V_C_SU = α×n×d×b_C
V_D_SO = V_C_SO = 0

C_gate_up = 4×n×d×m
C_proj = 2×n×d×m
F = F_peak×95%
T_gate_up = C_gate_up/F
T_proj = C_proj/F
```

V 的单位为 byte，F 为 operations/s，T 为秒；乘加计 2 次运算。只计单卡单向发送量，均衡接收量与之相等，收发不相加；gate/up 两分支不使 dispatch 流量翻倍。

本卡的 1/p 分配不经过网络，但仍参与 GEMM，故 α 只乘流量，不能同时乘到计算时间上抵消。E-FSDP 的参数分片和 E-DP 的副本数也不再缩减这里的单卡计算量。

## 有效带宽需求与标称链路需求

有效数据带宽需求为：

```text
B_D_req = V_D_SU/T_gate_up/10^9 = (p−1)/p × b_D×F/(4m×10^9)
B_C_req = V_C_SU/T_proj/10^9   = (p−1)/p × b_C×F/(2m×10^9)
```

单位 GB/s；t、K、d、E 均抵消。固定精度和算力后，需求由 m 与 EP Size 共同决定，**必须保留 `(p−1)/p`**。

| 精度组合 | Dispatch 有效需求（GB/s） | Combine 有效需求（GB/s） |
|---|---:|---:|
| FP8 | `(p−1)/p × 840275/m` | `(p−1)/p × 1680550/m` |
| BF16 | `(p−1)/p × 840275/m` | `(p−1)/p × 1680550/m` |
| DeepEP | `(p−1)/p × 840275/m` | `(p−1)/p × 3361100/m` |

FP8/BF16 的 b×F 相同，带宽需求相同。DeepEP dispatch 不变、combine 需求翻倍；相同场景下 FP8/BF16 的 combine 是 dispatch 的 2 倍，DeepEP 为 4 倍。

All2All 带宽利用率 η_A2A=70%。若将需求换成可选购/配置的标称链路带宽：

```text
B_D_link_req = B_D_req/0.70
B_C_link_req = B_C_req/0.70
```

结果同时保留有效需求和标称链路需求，避免混用单位口径。提高网络效率不改变有效数据需求，只改变需要的标称链路规格和掩盖率。

## SU 档位与分阶段掩盖率

本任务将六档 SU 数值解释为标称单卡单向带宽，再乘 70%；其他任务已有的有效带宽定义不变。

| 标称 SU 带宽（GB/s） | All2All 可用带宽（GB/s） |
|---:|---:|
| 400 | 280 |
| 600 | 420 |
| 800 | 560 |
| 1000 | 700 |
| 1200 | 840 |
| 1600 | 1120 |

令标称带宽为 L，V 使用 byte：

```text
T_comm_D = V_D_SU/(0.70×L×10^9)
T_comm_C = V_C_SU/(0.70×L×10^9)
rho_D = min(1,T_gate_up/T_comm_D) = min(1,0.70×L/B_D_req)
rho_C = min(1,T_proj/T_comm_C)    = min(1,0.70×L/B_C_req)
```

等价地，掩盖率为 `min(1,L/B_link_req)`；两种写法任选其一，不能重复折减 70%。零通信时掩盖率为 1。分项各自使用对应 GEMM 窗口，不能用两阶段平均时间替代；100% 表示理想稳态窗口覆盖，不包含启动和尾部。

## EP 整体掩盖率

整体指标为两阶段总通信时间中被各自 GEMM 掩盖的比例，按通信时间加权：

```text
rho_EP = [min(T_comm_D,T_gate_up)+min(T_comm_C,T_proj)]/(T_comm_D+T_comm_C)
       = (T_comm_D×rho_D+T_comm_C×rho_C)/(T_comm_D+T_comm_C)
T_exposed = (1−rho_EP)×(T_comm_D+T_comm_C)
```

两个阶段使用相同 SU 有效带宽，通信时间之比等于元素字节数之比，故不需要增加扫描因子：

| 场景 | 通信时间比 D:C | 整体掩盖率 |
|---|---:|---|
| FP8 / BF16 | 1:1 | `(rho_D+rho_C)/2` |
| DeepEP | 1:2 | `(rho_D+2×rho_C)/3` |

先对每个阶段的掩盖率截断到 1，再加权；不跨阶段借用剩余计算窗口，不能改用 `min(1,(T_gate_up+T_proj)/(T_comm_D+T_comm_C))`。零通信时整体掩盖率为 1；有通信时，整体为 100% 当且仅当两阶段均完全掩盖。该指标沿用理想分阶段模型，不表示实际调度后的 EP 总耗时。

## 输出与执行

[EP.xlsx](../../../results/EP.xlsx) 共 6 个页签：全部场景、FP8、BF16、DeepEP、中间维来源、说明。汇总页 189 行，各精度页 63 行，结果页均为 26 列：

| 字段 | 列数 / 含义 |
|---|---|
| `moe_expert_hidden_size`、`EP_Size`、`EP_precision_case` | 3；共同组成场景键 |
| `EP_remote_fraction` | 1；(p−1)/p |
| `EP_dispatch_su_bandwidth`、`EP_combine_su_bandwidth` | 2；有效数据带宽需求，GB/s |
| `EP_dispatch_su_link_bandwidth`、`EP_combine_su_link_bandwidth` | 2；按 70% 折算的标称链路带宽需求，GB/s |
| `EP_dispatch_overlap_ratio_su{L}GBps`、`EP_combine_overlap_ratio_su{L}GBps` | 12；六档各两项掩盖率，存储 0～1、显示百分比 |
| `EP_overall_overlap_ratio_su{L}GBps` | 6；追加在原 20 列末尾，按 400/600/800/1000/1200/1600 GB/s 排列；通信时间加权整体掩盖率，存储 0～1、显示百分比 |

不继承原集群场景 ID、HBM 标记或基础 21 列；按 m、p、精度三项关联到其他分析。t、K、d 已约去，仅凭本扫描无法唯一还原绝对流量/时间，因此不输出这些值。SO 恒为 0，在说明页注明，不重复增加结果列。

```bash
python3 scripts/generate_ep.py
python3 scripts/generate_ep.py --check
```

输入解析、计算、工作簿、独立核查分别位于 [ep_inputs.py](../../../scripts/su_so_modeling/ep_inputs.py)、[ep.py](../../../scripts/su_so_modeling/ep.py)、[ep_workbook.py](../../../scripts/su_so_modeling/ep_workbook.py)、[ep_validation.py](../../../scripts/su_so_modeling/ep_validation.py)。现有 `run_modeling.py` 继续管理十二份全量集群表，EP 维度扫描独立执行。

## 校验

核查完整笛卡尔积、无重复键、三个精度页与汇总页一致；使用独立的 token 载荷/GEMM 时间精确分数参考复算每项结果。另检查 95% 算力、70% 链路折减只应用一次、FP8/BF16 相等、DeepEP combine 翻倍、带宽随 m 反比/随 p 增加、掩盖率随 SU 档位非递减、比例边界及零通信。

整体掩盖率用原始通信时间与各 GEMM 窗口的 `sum(min)/sum` 独立复核，检查 DeepEP 的 1:2 权重、阶段饱和后不借用窗口，以及整体完全掩盖与两个分项均完全掩盖等价。扩展时逐页比较原 20 列，数值和顺序保持一致。

本轮仅前向；反向的激活梯度精度和 dX/dW 窗口单独建模。
