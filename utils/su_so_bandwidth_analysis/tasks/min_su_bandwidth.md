# 各模型最小 SU 带宽配置

在现有扫描范围内，对每个模型、每种精度分别寻找使最大分项 SU 需求最低的配置。输出 [SU最小带宽配置.xlsx](../../../results/SU最小带宽配置.xlsx)，独立于原任务结果表。

## 输入与范围

- [模型输入](../../../max_layer_parameters.json)、[场景输入](../scenario_inputs.md)；沿用全部拓扑、并行约束和 HBM 检查。
- FSDP、E-FSDP、E-DP 仅使用分层通信；EP 沿用 SU 内前向 All2All；Engram 沿用所选层按需查表。
- FP8、BF16、DeepEP 分别优化。DeepEP 的 HBM、参数通信和整层计算使用 FP8 配置，仅 EP Combine 通信使用 BF16。
- 192/288 GB HBM 均可选，同时单独记录两种 HBM 的最优结果。
- 不限定总卡数或 SU Size 为固定值，不施加 SO 100/200 GB/s 上限，也不优化总成本或吞吐率。

## 目标与带宽口径

对一个 HBM 合法配置 x，定义：

```text
B_SU(x) = max(
  B_FSDP_fwd_SU, B_FSDP_bwd_SU,
  B_EFSDP_fwd_SU, B_EFSDP_bwd_SU, B_EDP_bwd_SU,
  B_EP_dispatch_effective/0.70, B_EP_combine_effective/0.70,
  B_Engram_fwd_5layers_SU, B_Engram_bwd_1layer_SU
)
x_best = argmin_x B_SU(x)
```

EP 按已确认的 70% All2All 利用率折算到标称档位；其他任务沿用现有档位直接作为有效带宽的假设，不额外折减。结果称为 **SU 配置带宽需求**，单位 GB/s，不将 EP 有效需求与其他档位需求直接混用。EP GEMM 为 95%，整层计算为 40%。

目标要求已建模的各分项独立达到 100% 掩盖，EP 两阶段均须满足；未加入 EP 反向，也不叠加各任务共享链路的争用，因此这是独立掩盖口径下的带宽下界。

Engram采用扫描中最短的前向1层、反向1层窗口，覆盖其他更长窗口；不靠额外延长窗口降低本次目标。SO需求同样取各已建模分项的最大值，仅记录作诊断，不用于筛选。

若最小带宽对应多个配置，依次选择总卡数较小、SU Size 较小、HBM 规格较小、EP 较小、E-FSDP 较小、E-DP 较小、场景 ID 较小的配置。FSDP 始终等于总卡数 N。

## 输出与执行

工作簿有 7 页：模型汇总、最优配置、FP8、BF16、DeepEP、分HBM规格、说明。当前 16 个模型对应 48 行最优配置、96 行分 HBM 最优配置；每种精度页 16 行。明细 24 列包含场景 ID、拓扑、并行分组、HBM 占用、SU 最小需求及匹配档位、瓶颈项、各任务 SU 需求、SO 诊断值。

模型汇总的策略简写为 `SU Size / EP / E-FSDP / E-DP`，`N=EP×E-FSDP×E-DP`。DeepEP 的场景 ID 对应原 FP8 基础行。最小档位超过 1600 GB/s 时明确标记，不截断结果。

```bash
python3 scripts/analyze_min_su_bandwidth.py
python3 scripts/analyze_min_su_bandwidth.py --check
```

代码：[计算与独立复核](../../../scripts/su_so_modeling/min_su_bandwidth.py)、[导出入口](../../../scripts/analyze_min_su_bandwidth.py)。完整枚举与独立约数计数一致；所选配置按原始通信量和精确计算时间复核 HBM、分组及全部带宽；工作簿逐页校验后发布。
