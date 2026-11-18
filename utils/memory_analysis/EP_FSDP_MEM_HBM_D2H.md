## 实验设定

- 配置文件：`configs/` 目录下带有 [`qwen_`, `dsv`, `glm_`, `hy4_`, `kimi_`, `qwen`] 前缀的配置文件
- 算力规格：970DT — BF16 1769 Tops，FP8 3538 Tops，FP4 14155 Tops, 980DT — BF16 3538 Tops，FP8 7076 Tops，FP4 28310 Tops,
- HBM 容量：[96, 120, 144, 168, 192, 240, 288]GB七种情况，对应[4,5,6,7,8,10,12]颗ZQ存储介质，每颗24GB；逐模型报告对七种容量分别计算
- MFU：0.40
- weight-dtype: {bf16, fp8}两种情况
- activation-dtype: {bf16, fp8}两种情况
- 精度组合按笛卡尔积运行，共四组：bf16/bf16、bf16/fp8、fp8/bf16、fp8/fp8
- FP8 权重组合的 D2H 掩盖计算使用硬件 FP8 峰值算力；激活 dtype 控制激活显存，不改变本次计算峰值选择
- 每组报告和 PPT 单独写入 `reports_analyse/ep_fsdp_mem_hbm_d2h/weight_{weight_dtype}_activation_{activation_dtype}/`；PPT包含原定最佳策略场景HBM=288GB/DDR=768GB和HBM=192GB/DDR=512GB，并额外筛选HBM七档均使用DDR=512GB时的容量满足情况。PPT考虑下在HBM=[96, 120, 144, 168, 192, 240, 288]GB七种情况，DDR=512能否满足训练要求。

假设训练所需卡数K=12288，遍历以下并行策略，执行任务一、二、三、四：
EP=8, EDP=floor(K/8), EFSDP=1, FSDP=K
EP=8, EDP=floor(K/8/2), EFSDP=2, FSDP=K
EP=8, EDP=floor(K/8/4), EFSDP=4, FSDP=K
EP=16, EDP=floor(K/16), EFSDP=1, FSDP=K
EP=16, EDP=floor(K/16/2), EFSDP=2, FSDP=K
EP=16, EDP=floor(K/16/4), EFSDP=4, FSDP=K
EP=16, EDP=floor(K/16/8), EFSDP=8, FSDP=K
EP=32, EDP=floor(K/32), EFSDP=1, FSDP=K
EP=32, EDP=floor(K/32/2), EFSDP=2, FSDP=K
EP=32, EDP=floor(K/32/4), EFSDP=4, FSDP=K
EP=32, EDP=floor(K/32/8), EFSDP=8, FSDP=K
EP=32, EDP=floor(K/32/16), EFSDP=16, FSDP=K
EP=32, EDP=floor(K/32/32), EFSDP=32, FSDP=K
EP=64, EDP=floor(K/64), EFSDP=1, FSDP=K
EP=64, EDP=floor(K/64/2), EFSDP=2, FSDP=K
EP=64, EDP=floor(K/64/4), EFSDP=4, FSDP=K
EP=64, EDP=floor(K/64/8), EFSDP=8, FSDP=K
EP=64, EDP=floor(K/64/16), EFSDP=16, FSDP=K
EP=64, EDP=floor(K/64/32), EFSDP=32, FSDP=K
EP=128,EDP=floor(K/128), EFSDP=1, FSDP=K
EP=128,EDP=floor(K/128/2), EFSDP=2, FSDP=K
EP=128,EDP=floor(K/128/4), EFSDP=4, FSDP=K
EP=128,EDP=floor(K/128/8), EFSDP=8, FSDP=K
EP=128,EDP=floor(K/128/16), EFSDP=16, FSDP=K
EP=128,EDP=floor(K/128/32), EFSDP=32, FSDP=K
EP=256,EDP=floor(K/256), EFSDP=1, FSDP=K
EP=256,EDP=floor(K/256/2), EFSDP=2, FSDP=K
EP=256,EDP=floor(K/256/4), EFSDP=4, FSDP=K
EP=256,EDP=floor(K/256/8), EFSDP=8, FSDP=K
EP=256,EDP=floor(K/256/16), EFSDP=16, FSDP=K
EP=256,EDP=floor(K/256/32), EFSDP=32, FSDP=K
EP=512,EDP=floor(K/512), EFSDP=1, FSDP=K
EP=512,EDP=floor(K/512/2), EFSDP=2, FSDP=K
EP=512,EDP=floor(K/512/4), EFSDP=4, FSDP=K
EP=512,EDP=floor(K/512/8), EFSDP=8, FSDP=K
EP=512,EDP=floor(K/512/16), EFSDP=16, FSDP=K
EP=512,EDP=floor(K/512/32), EFSDP=32, FSDP=K
EP=1024,EDP=floor(K/1024), EFSDP=1, FSDP=K
EP=1024,EDP=floor(K/1024/2), EFSDP=2, FSDP=K
EP=1024,EDP=floor(K/1024/4), EFSDP=4, FSDP=K
EP=1024,EDP=floor(K/1024/8), EFSDP=8, FSDP=K
EP=1024,EDP=floor(K/1024/16), EFSDP=16, FSDP=K
EP=1024,EDP=floor(K/1024/32), EFSDP=32, FSDP=K
EP=2048,EDP=floor(K/2048), EFSDP=1, FSDP=K
EP=2048,EDP=floor(K/2048/2), EFSDP=2, FSDP=K
EP=2048,EDP=floor(K/2048/4), EFSDP=4, FSDP=K
EP=2048,EDP=floor(K/2048/8), EFSDP=8, FSDP=K
EP=2048,EDP=floor(K/2048/16), EFSDP=16, FSDP=K
EP=2048,EDP=floor(K/2048/32), EFSDP=32, FSDP=K

## 输入

- `seq_len` = 4096
- `batch_size` = 1

## 输出形式

每个模型生成一份 Markdown 报告，结构如下：

```
## {模型名} ({总参数} params) — K={卡数}

### HBM={96|120|144|168|192|240|288}GB

#### 任务一: DDR容量诉求 / 任务二: D2H带宽诉求

| EP | FSDP=K | EDP | EFSDP | 单卡MoE(GB) | 单卡BB(GB) | 静态HBM(GB) | HBM剩余(GB) | DDR诉求(GB) | MoE卸载% | 全卸载% | D2H-MoE(GB/s) | D2H-Full(GB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EP=128 | 64,374 | 251 | 251 | ... | ... | ... | +78.7 ✓ | +824.1 ⚠️ | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

注：HBM剩余为负标 ⚠️，DDR诉求为正标 ⚠️。

#### 任务三: D2H带宽遍历 → 端到端收益 (不考虑DDR)

| EP | FSDP=K | EDP | EFSDP | 200G→160G | 250G→200G | ... | 500G→400G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EP=128 | 64,374 | 251 | 251 | 5.59% | 6.99% | ... | 13.98% |
| ... | ... | ... | ... | ... | ... | ... | ... |

#### 任务四: DDR容量遍历 → 端到端收益 (不考虑D2H带宽)

| EP | FSDP=K | EDP | EFSDP | 优化器+主权重(GB) | DDR=256GB | ... | DDR=768GB |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| EP=128 | 64,374 | 251 | 251 | 576.7 | 0.00% (0%) | ... | 7.36% (29%) |
| ... | ... | ... | ... | ... | ... | ... | ... |

注：括号内为 offload_ratio，端到端收益 = offload_ratio × 25%。
```

## 内存建模方法

约束条件：`SU=K`，`EP < SU`，`EDP=floor(K/EP/EFSDP)`，`EFSDP`，`FSDP=SU`。

使用如下命令获取模型结构数据：

```
python3 -m src.main --config configs/{config} --seq-len {seq_len} --batch-size {batch_size} --weight-dtype {weight-dtype} --activation-dtype {activation-dtype} --format json
```

批量生成报告与 PPT 的命令示例：

```
python3 -m utils.memory_analysis.generate_ep_fsdp_reports --weight-dtype fp8 --activation-dtype bf16
python3 -m utils.memory_analysis.generate_best_strategy_ppt --weight-dtype fp8 --activation-dtype bf16
```

### 静态 HBM 占用

```
单卡MoE部分占用HBM     = MoE(weight_dtype) / EP / EFSDP * 2
单卡BackBone部分占用HBM = Backbone(weight_dtype) / FSDP * 2
静态HBM占用 = 最大一层激活显存占用 + 单卡MoE部分占用HBM + 单卡BackBone部分占用HBM + 3 GB
```

> 最大一层激活显存占用归入"静态HBM"是简化处理 — 实际训练中激活值随 micro-batch 变化，此处近似认为峰值激活常驻。

### 激活显存

```
全激活占用空间    = sum(第i层的激活值占用 for i in range(N))
MoE激活占用空间   = sum(第i层MoE的激活值占用 for i in range(N))
HBM剩余          = HBM容量 * 0.95 - 静态HBM占用
```

### 优化器与 DDR

```
优化器与主权重占用空间 = (MoE(FP32) / EP / EFSDP + Backbone(FP32) / FSDP) * 2
DDR容量诉求（全卸载） = 优化器与主权重占用空间 + 全激活占用空间 - HBM剩余
DDR容量诉求（仅卸载MoE） = 优化器与主权重占用空间 + MoE激活占用空间 - HBM剩余

```

### 卸载比例

```
MoE激活卸载比例  = min(1.0, max(0, (MoE激活占用空间 - HBM剩余) / MoE激活占用空间))
全激活卸载比例   = min(1.0, max(0, (全激活占用空间 - HBM剩余) / 全激活占用空间))
```

### 层计算耗时

```
第i层MoE激活显存占用 = batch_size * seq_len * (top_k + num_shared_experts) 个 token 对应的一个专家 FFN 激活
第i层MoE计算量       = 单个专家的单token计算量 * batch_size * seq_len * (top_k + num_shared_experts)
第i层的计算耗时       = (第i层Attention计算量 + 第i层MoE计算量) / (硬件Flops * MFU)
第i层的卸载耗时       = 第i层激活显存占用 * 全激活卸载比例 / D2H带宽
```

> shared expert 处理所有 token，routed expert 每个 token 仅激活 top_k 个，有效专家数 = top_k + num_shared_experts。

### D2H 带宽诉求

```
D2H带宽诉求1 (只卸载MoE)   = max( 第i层MoE激活显存占用 * MoE激活卸载比例   / 第i+1层前向总耗时   for i in range(N-1) )
D2H带宽诉求2 (全激活卸载)  = max( 第i层激活显存占用     * 全激活卸载比例    / 第i+1层前向总耗时   for i in range(N-1) )
```

> D2H 掩盖窗口：第 *i* 层卸载被第 *i+1* 层计算掩盖（*i* = 0 .. N-2）。第 0 层没有前置计算可掩盖卸载，最后一层的卸载无后续计算可掩盖 — 这两个边缘场景在 `max over N-1 layers` 中已隐含处理。


### SSD 容量诉求



### 任务三: D2H 带宽遍历

带宽范围 `[200, 250, 300, 350, 400, 450, 500]` GB/s，利用率 80%，实际带宽 `[160, 200, 240, 280, 320, 360, 400]` GB/s。

思路：在全卸载场景下 D2H 无法完全被计算掩盖；通过调整卸载比例使 D2H 完全被计算掩盖。

```
if 下一层的计算耗时 - 当前层的卸载耗时 >= 0 → cover_ratio = 100%
if 下一层的计算耗时 - 当前层的卸载耗时 <  0 → cover_ratio = 下一层的计算耗时 / 当前层的卸载耗时
```

重计算在端到端训练时间中占比为 25%（推导：前向 2 FLOPs，反向 4 FLOPs，重计算 2 FLOPs → 2 / (2+4+2) = 2/8 = 25%）：

```
端到端收益 = cover_ratio × 25%
```

### 任务四: DDR 容量遍历

DDR 容量范围 `[256, 384, 512, 540, 768]` GB。

```
可用DDR空间            = 总DDR空间 - 优化器与主权重占用空间
激活需要放在DDR上的容量 = 全激活占用空间 - HBM剩余

if 可用DDR空间 >= 激活需要放在DDR上的容量 → offload_ratio = 100%
if 可用DDR空间 <  激活需要放在DDR上的容量 → offload_ratio = 1 - (激活需要放在DDR上的容量 - 可用DDR空间) / 激活需要放在DDR上的容量
```

重计算在端到端训练时间中占比为 25%：

```
端到端收益 = offload_ratio × 25%
```


### 任务五：SSD容量

SSD用于存储训练 checkpoint。此处统计**每 GPU 的 checkpoint 分片容量**，并非全作业的 checkpoint 总容量，也不是 SSD 硬件规划容量。

```
单卡MoE checkpoint分片
  = (MoE权重 + 对应的master weights + 对应的optimizer states) / EP / EFSDP

单卡Backbone checkpoint分片
  = (Backbone权重 + 对应的master weights + 对应的optimizer states) / FSDP

单卡SSD分片需求
  = 单卡MoE checkpoint分片 + 单卡Backbone checkpoint分片
```

使用配置中各模型实际的 `weights`、`master_weights` 和 `optimizer_states` 字节数；checkpoint 估算不包含训练梯度。模型未提供按参数组拆分的 optimizer 状态时，按对应权重比例估算。整机/节点 SSD 需求还需乘以每节点 GPU 数；全作业逻辑 checkpoint 总容量应以模型全局权重、master weights 和 optimizer states 汇总，不能将单卡数直接当作全局总量。


## 任务

参考建模方法章节，执行下列任务：

1. **任务一**：输出 DDR 容量诉求(全卸载)， 输出 DDR 容量诉求(仅卸载MoE)
2. **任务二**：输出 D2H 带宽诉求
3. **任务三**：不考虑 DDR 容量不够的情况下，遍历带宽，输出对应带宽下消除重计算比例（端到端收益）
4. **任务四**：不考虑 D2H 带宽不够的情况下，遍历 DDR 容量，输出对应 DDR 容量下消除重计算比例（端到端收益）
5. **任务五**: 输出SSD容量诉求


注意：所有折线图均添加数据标签，数据标签仅保留0位小数。
最佳并行策略是EFSDP=1，考虑在EP最小的情况，并且HBM有剩余空间。
1. 在HBM=288GB，DDR=768GB时候，给出各个模型的最佳并行策略；在HBM=192GB，DDR=512GB时候，给出各个模型的最佳并行策略。做成表格的形式。
2. 给出在上述并行策略下，各个模型的DDR容量诉求，做成PPT插入图表的折线图。
3. 给出在上述并行策略下，各个模型的SSD容量诉求，做成PPT插入图表的折线图。
4. 给出在上述并行策略下，各个模型的D2H带宽诉求，做成PPT的表格，和折线图，并且给出在D2H=200GB/s时（考虑带宽利用率在80%下）的消除重计算的比例（也即端到端收益），绘制成PPT的表格。
5. 给出在上述并行策略下，各个模型随着D2H带宽增加，统计能消除重计算的模型个数，统计消除重计算的平均值，绘制成PPT表格。
6. 给出在上述并行策略下，各个模型随着DDR容量增加，统计能消除重计算的模型个数，统计消除重计算的平均值，绘制成PPT表格。
7. 在HBM=96/120/144/168/192/240/288GB各档位，按DDR=512GB场景分别选出最佳并行策略，展示各模型MoE激活及全激活的D2H带宽诉求折线图。
8. 在HBM=96/120/144/168/192/240/288GB各档位，按DDR=512GB场景的策略选择规则给出最佳并行策略，并用表格列出HBM余量、DDR全卸载及仅MoE卸载容量诉求、D2H-MoE及D2H-Full带宽诉求，以及D2H=200GB/s（80%利用率）和DDR=512GB时的端到端收益；另以折线图展示HBM余量、DDR全卸载诉求和两种收益随模型及HBM档位的变化。Qwen3.5因缺少匹配配置不纳入数值折线图。
9. 在HBM=96/120/144/168/192/240/288GB各档位，按对应档位策略计算D2H=200GB/s（80%利用率，有效带宽160GB/s）时的端到端收益，不考虑DDR容量限制；以表格列出模型与各档位收益，并以折线图展示收益随HBM档位的变化。Qwen3.5因缺少匹配配置，在表中标记为未计算且不纳入数值折线图。
9. 给出在HBM=96/120/144/168/192/240/288GB各档位，不考虑DDR容量不够的情况下，200GB/s的D2H带宽，端到端收益，给出折线图和表格。