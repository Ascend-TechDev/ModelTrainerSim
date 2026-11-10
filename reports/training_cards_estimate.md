# 大模型训练卡数估算

目标：让各模型在 **1~2 个月** 内完成训练，估算 950DT、960DT、970DT、980DT 各代硬件所需卡数。
横轴按模型总参数量从小到大排列。

## 1. 计算口径

```text
训练 FLOPs  = 6 × 激活参数量 × 训练 token 数
训练 token 数 D = 10 × 模型总参数量
所需卡数 N = 训练 FLOPs ÷ (目标时间 T × 单卡 BF16 峰值 × MFU)
```

- 1 个月按 **30 天**（2,592,000 s），2 个月按 **60 天**（5,184,000 s）
- **MFU = 35%**；卡数与 MFU 成反比（MFU 30% 时 ×1.17，40% 时 ×0.875）
- 卡数向上取整
- 纯算力 roofline，不建模通信、显存、断点续训、故障恢复和负载不均

## 2. 硬件 BF16 峰值

| 代际 | 单卡 BF16 峰值 | 说明 |
|---|---:|---|
| 950DT | 432 TFLOPS | 对应 `hardware/` 中 A5 机型 |
| 960DT | 943 TFLOPS | 对应 `hardware/` 中 DV122 机型 |
| 970DT | 1,769 TFLOPS | 970DT / 970PR 峰值相同 |
| 980DT | 3,538 TFLOPS | 按 970DT 的 2 倍假设（`hardware/` 中暂无配置） |

> 950DT / 960DT 分别对应 `hardware/` 中的 A5 与 DV122 机型，本图按给定峰值 **432 / 943 TFLOPS** 计算（不是 A5=487、DV122=1000 的 `cube_fp16`）；970DT / 980DT 按 1769 / 3538 TFLOPS。

## 3. 模型规模与训练数据

| 模型 | 家族 | 总参数 | 激活参数/token | 训练数据 D（10 × 总参数） |
|---|---|---:|---:|---:|
| Qwen3.5-397B | Qwen | 0.397 T | 17.00 B | 4.0 T tokens |
| GLM-5.2 | GLM | 0.743 T | 41.25 B | 7.4 T tokens |
| DSV41Flash | DeepSeek | 0.748 T | 16.13 B | 7.5 T tokens |
| DSV4Pro | DeepSeek | 1.574 T | 49.39 B | 15.7 T tokens |
| Kimi-10T-D | Kimi | 9.283 T | 155.06 B | 92.8 T tokens |
| Hy4-10T | Hunyuan | 10.074 T | 342.57 B | 100.7 T tokens |
| GLM-10T | GLM | 11.295 T | 145.30 B | 112.9 T tokens |
| Kimi-10T-W | Kimi | 11.918 T | 248.80 B | 119.2 T tokens |
| DS-10T | DeepSeek | 15.835 T | 140.02 B | 158.4 T tokens |
| DS-20T | DeepSeek | 31.656 T | 213.13 B | 316.6 T tokens |
| Hy4-30T | Hunyuan | 41.501 T | 344.75 B | 415.0 T tokens |
| Kimi-50T | Kimi | 49.996 T | 684.15 B | 500.0 T tokens |
| DS-40T | DeepSeek | 63.269 T | 354.19 B | 632.7 T tokens |
| Kimi-100T | Kimi | 105.003 T | 1122.36 B | 1050.0 T tokens |
| DS-80T | DeepSeek | 126.484 T | 474.48 B | 1264.8 T tokens |

> 说明：
> - DeepSeek DS 系列名称不代表实际总参数量，按配置计算为 15.8 / 31.7 / 63.3 / 126.5 T。
> - Kimi 包含 10T deep/wide、50T、100T 四个仓内变体。
> - **Qwen3.5-397B-A17B 为外部数据点**：ModelTrainerSim 的 Qwen 实现仍是占位，总参 397B、激活 17B 取自官方模型命名；卡数仅供参考。
> - GLM-5.2 按仓内最新修正后的 743B 参数口径。
> - GLM-10T：100 层、2048 路由专家（18 激活）+ 2 共享、moe_inter 3072、128 heads、vocab 154880。
> - Hy4-30T：120 层（1 dense + 119 MoE）、4096 路由专家（24 激活）+ 2 共享、moe_inter 4608、160 heads、vocab 120832。

## 4. 所需卡数（训练 1 个月）

| 模型 | 家族 | 总参数 | 激活/token | 训练数据 D | 950DT | 960DT | 970DT | 980DT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-397B | Qwen | 0.397 T | 17.00 B | 4.0 T tok | 1,034 | 474 | 253 | 127 |
| GLM-5.2 | GLM | 0.743 T | 41.25 B | 7.4 T tok | 4,695 | 2,151 | 1,147 | 574 |
| DSV41Flash | DeepSeek | 0.748 T | 16.13 B | 7.5 T tok | 1,849 | 847 | 452 | 226 |
| DSV4Pro | DeepSeek | 1.574 T | 49.39 B | 15.7 T tok | 11,898 | 5,451 | 2,906 | 1,453 |
| Kimi-10T-D | Kimi | 9.283 T | 155.06 B | 92.8 T tok | 220,368 | 100,953 | 53,815 | 26,908 |
| Hy4-10T | Hunyuan | 10.074 T | 342.57 B | 100.7 T tok | 528,344 | 242,041 | 129,025 | 64,513 |
| GLM-10T | GLM | 11.295 T | 145.30 B | 112.9 T tok | 251,252 | 115,102 | 61,358 | 30,679 |
| Kimi-10T-W | Kimi | 11.918 T | 248.80 B | 119.2 T tok | 453,944 | 207,957 | 110,856 | 55,428 |
| DS-10T | DeepSeek | 15.835 T | 140.02 B | 158.4 T tok | 339,463 | 155,512 | 82,899 | 41,450 |
| DS-20T | DeepSeek | 31.656 T | 213.13 B | 316.6 T tok | 1,032,912 | 473,190 | 252,243 | 126,122 |
| Hy4-30T | Hunyuan | 41.501 T | 344.75 B | 415.0 T tok | 2,190,437 | 1,003,467 | 534,918 | 267,459 |
| Kimi-50T | Kimi | 49.996 T | 684.15 B | 500.0 T tok | 5,236,670 | 2,398,984 | 1,278,826 | 639,413 |
| DS-40T | DeepSeek | 63.269 T | 354.19 B | 632.7 T tok | 3,430,719 | 1,571,655 | 837,802 | 418,901 |
| Kimi-100T | Kimi | 105.003 T | 1122.36 B | 1050.0 T tok | 18,042,552 | 8,265,517 | 4,406,096 | 2,203,048 |
| DS-80T | DeepSeek | 126.484 T | 474.48 B | 1264.8 T tok | 9,187,937 | 4,209,108 | 2,243,748 | 1,121,874 |

## 5. 所需卡数（训练 2 个月）

| 模型 | 家族 | 总参数 | 激活/token | 训练数据 D | 950DT | 960DT | 970DT | 980DT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-397B | Qwen | 0.397 T | 17.00 B | 4.0 T tok | 517 | 237 | 127 | 64 |
| GLM-5.2 | GLM | 0.743 T | 41.25 B | 7.4 T tok | 2,348 | 1,076 | 574 | 287 |
| DSV41Flash | DeepSeek | 0.748 T | 16.13 B | 7.5 T tok | 925 | 424 | 226 | 113 |
| DSV4Pro | DeepSeek | 1.574 T | 49.39 B | 15.7 T tok | 5,949 | 2,726 | 1,453 | 727 |
| Kimi-10T-D | Kimi | 9.283 T | 155.06 B | 92.8 T tok | 110,184 | 50,477 | 26,908 | 13,454 |
| Hy4-10T | Hunyuan | 10.074 T | 342.57 B | 100.7 T tok | 264,172 | 121,021 | 64,513 | 32,257 |
| GLM-10T | GLM | 11.295 T | 145.30 B | 112.9 T tok | 125,626 | 57,551 | 30,679 | 15,340 |
| Kimi-10T-W | Kimi | 11.918 T | 248.80 B | 119.2 T tok | 226,972 | 103,979 | 55,428 | 27,714 |
| DS-10T | DeepSeek | 15.835 T | 140.02 B | 158.4 T tok | 169,732 | 77,756 | 41,450 | 20,725 |
| DS-20T | DeepSeek | 31.656 T | 213.13 B | 316.6 T tok | 516,456 | 236,595 | 126,122 | 63,061 |
| Hy4-30T | Hunyuan | 41.501 T | 344.75 B | 415.0 T tok | 1,095,219 | 501,734 | 267,459 | 133,730 |
| Kimi-50T | Kimi | 49.996 T | 684.15 B | 500.0 T tok | 2,618,335 | 1,199,492 | 639,413 | 319,707 |
| DS-40T | DeepSeek | 63.269 T | 354.19 B | 632.7 T tok | 1,715,360 | 785,828 | 418,901 | 209,451 |
| Kimi-100T | Kimi | 105.003 T | 1122.36 B | 1050.0 T tok | 9,021,276 | 4,132,759 | 2,203,048 | 1,101,524 |
| DS-80T | DeepSeek | 126.484 T | 474.48 B | 1264.8 T tok | 4,593,969 | 2,104,554 | 1,121,874 | 560,937 |

## 6. 图表

### 6.1 训练 1 个月

![训练 1 个月所需卡数](training_cards_estimate.png)

- PNG：[training_cards_estimate.png](training_cards_estimate.png)
- SVG：[training_cards_estimate.svg](training_cards_estimate.svg)

### 6.2 训练 2 个月

![训练 2 个月所需卡数](training_cards_estimate_2m.png)

- PNG：[training_cards_estimate_2m.png](training_cards_estimate_2m.png)
- SVG：[training_cards_estimate_2m.svg](training_cards_estimate_2m.svg)

生成脚本：[plot_training_cards.py](plot_training_cards.py)

## 7. 结论与注意事项

1. **硬件代际比例**：同样任务、同样 MFU 下，卡数约为
   `950DT : 960DT : 970DT : 980DT = 8.19 : 3.75 : 2 : 1`。
   970DT 约为 960DT 的 53%、950DT 的 24%；980DT 正好是 970DT 的一半。
2. **MFU 敏感性**：卡数 ∝ 1/MFU。MFU 取 30% 时上表 ×1.17，取 40% 时 ×0.875。
3. **10×总参对 MoE 仍然很激进**：DS-80T 的 D = 1.26 P tokens，即使 980DT 也要约 112 万卡才能一个月跑完。若改用“10×激活参数”口径，卡数会按 总参/激活 的比例下降（DS-80T 约 267 倍），结果会现实得多。两种口径差别很大，汇报时建议明确写清楚。
4. **模型分布**：小模型端 Qwen3.5-397B、GLM-5.2、DSV41Flash 在千卡量级；GLM-10T（11.3T）约 25 万卡（950DT 一个月），Hy4-30T（41.5T）约 219 万卡；Kimi-100T 约 1,800 万卡、DS-80T 约 920 万卡（均为 950DT 一个月）。
5. **Qwen 数据为外部估算**：仓库内 Qwen 仅有占位实现，未生成完整报告；397B/17B 取自官方命名，卡数仅作横向参考。
6. **这是纯算力估算**：万卡以上集群的端到端 MFU 通常低于 35%，实际所需卡数会更高。
7. **980DT 为假设硬件**：目前只按 BF16 = 2 × 970DT 参与计算；若要落成硬件配置，还需补充 pod 规模、显存、带宽和互联参数。

## 8. 数据来源

- 模型参数与激活参数：`reports/*_report.json`（由 `configs/*.json` 生成）
- Qwen3.5-397B-A17B：官方模型命名（397B-A17B）；仓库 `src/models/qwen` 仍为 placeholder
- 硬件 BF16 峰值：950DT / 960DT 按给定 432 / 943 TFLOPS；970DT 1769 TFLOPS 取自 `hardware/` 970 配置；980DT 按 2 × 970DT 假设
- 图表与脚本：`reports/training_cards_estimate.png` / `.svg` / `training_cards_estimate_2m.*` / `plot_training_cards.py`
