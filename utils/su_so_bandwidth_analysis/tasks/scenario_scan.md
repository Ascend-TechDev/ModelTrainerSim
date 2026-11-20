# 全场景扫描

状态：已实现。代码：[scenario_scan.py](../../../scripts/su_so_modeling/scenario_scan.py)。

## 输入

- [模型输入](../model_inputs.md)：模型 ID、名称、所选层 ID、专家总数 E、Top-K。
- [场景输入](../scenario_inputs.md)：训练精度、HBM、SU Size、k、EP 候选，以及算力和并行约束。

## 方法

按以下顺序枚举，保持场景 ID 稳定：

```text
模型 → S → k → 精度 → HBM → p → F
N = S×k < 200000
p < S；S % p = 0；E % p = 0
F = N/p 的全部正整数约数，按升序
R = N/(p×F)
```

FSDP=N。N、FSDP、R 是推导量，不增加扫描维度。不同 S、k 即使 N 相同也分别保留。

## 输出

`iter_scenarios(models, settings)` 依次产生 18 列基础场景：

```text
场景ID、模型ID、模型名称、训练精度、单卡HBM (GB)、
SU Size、SU域数量 k、SO Size、FSDP Size、EP Size、E-FSDP Size、E-DP Size、
专家数 E、激活专家数 K、所选MoE层ID、峰值算力 (TOPS)、算力利用率、有效算力 (TOPS)
```

任务本身不按 HBM 容量删行。与 HBM 检查组合后，输出根目录 [SU_SO场景总表.xlsx](../../../SU_SO场景总表.xlsx) 的基础 21 列，各任务表复用同一基础。各模型页沿用汇总页的场景 ID。

## 校验

使用独立的质因数分解约数计数核查场景数量；检查候选范围、规模乘积、整除条件、顺序及唯一性。当前数量见 [结果核查](../validation.md)。
