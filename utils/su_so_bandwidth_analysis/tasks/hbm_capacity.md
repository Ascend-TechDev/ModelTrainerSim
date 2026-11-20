# HBM 容量检查

状态：已实现。代码：[hbm_capacity.py](../../../scripts/su_so_modeling/hbm_capacity.py)。

## 输入

- [基础场景](scenario_scan.md)：N、p、F、R、训练精度、HBM 容量。
- [模型输入](../model_inputs.md)：整模型非专家、路由专家及各张 Engram 表参数。
- [场景输入](../scenario_inputs.md)：四类训练状态字节数、HBM 占用上限。

## 方法

参数量换算为个参数：

```text
P_non = 10^9 × (model_non_expert_params_b + Σ model_engram_layer_<id>_params_b)
P_route = 10^9 × model_routed_expert_total_params_b

M_GB = b_state × [P_non/N + P_route/(p×F)] / 10^9
     = b_state × (P_non + R×P_route) / (N×10^9)
HBM占用比例 = M_GB / H_card
是否合法 = 1，当 HBM占用比例 ≤ hbm_limit_ratio；否则为 0
```

非专家含全部 Engram 表；路由专家按完整参数存储，不按 Top-K 缩减。E-DP 增加副本，单卡占用不再除以 R。仅计权重、梯度、单份 Muon 状态、主权重。

## 输出

`capacity_metrics(model, N, R, b_state, H_card, hbm_limit_ratio)` 返回：

| 字段 | 含义 |
|---|---|
| 是否合法 | 1/0；边界相等为合法 |
| 单卡HBM占用 (GB) | 单卡四类训练状态占用 |
| HBM占用比例 | 占完整 HBM 的比例 |

上述 3 列接在场景扫描 18 列之后，组成 [SU_SO场景总表.xlsx](../../../SU_SO场景总表.xlsx)，也作为所有任务表的公共基础。全部场景保留；该标记只表示基础存储约束，可行方案分析筛选 1。

## 校验

参数通过十进制还原为整数；容量合法性使用整数交叉乘法比较。导出后以两个分片分母的精确分数公式独立复核，不使用显示舍入值判断。
