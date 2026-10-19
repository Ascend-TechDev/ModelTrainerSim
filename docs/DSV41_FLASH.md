# DeepSeek V4.1 Flash 第一版分析

本实现依据 `src/models/dsv41flash/config.json` 中保存的本地官方格式配置快照，以及对应 `inference/model.py` 的张量形状。运行时不依赖原始模型目录，不加载权重，不需要 PyTorch 或 GPU。

## 运行

在仓库根目录执行：

```powershell
# 默认：40 层文本主干 + 两个 Engram，B=1、4K；BF16 逻辑权重
python -m src.main --config configs/dsv41flash_config.json

# 8K 和完整 JSON：含每个张量的 shape、参数组、每层结果、缓存来源
python -m src.main --config configs/dsv41flash_config.json --seq-len 8192 --format json --output flash_8k.json

# 32 卡场景：GPU 上 dense/expert 分片，CPU 上 Engram 大表和状态
python -m src.main --config configs/dsv41flash_32gpu_config.json

# 无额外依赖的验证
python -m unittest tests.test_dsv41flash -v
```

所有 Flash 文本报告中的 GB 均为 `1e9` 字节。JSON 保存原始 bytes。原 Pro 配置和计算公式保持原有行为。

Flash 与 Pro 共用同一套六段文本报告：Configuration、Parameters、Compute、Memory、Per-Module Summary、Per-Layer Summary，标题、字段和表格列宽一致。Flash 的容量继续使用十进制 GB；Pro 保留原有容量换算口径。

MoE 在展示表中拆为 `gate / routedexpert / sharedexpert`，每行 Count 表示层数。共用输入输出缓冲只在 gate 行计一次；专家中间量按路由/共享专家数分摊，三行加总与原始 MoE 相同。`csa_full / csa_reindex / csa_reuse` 保留 Flash 的结构区别。JSON 的原始模块划分和计算值不变。

默认报告省略参数组长表，详细假设通过 JSON 查看。32 卡或 CPU 放置场景追加每设备静态容量；启用 DSpark 时追加独立草稿计算摘要。逐层表展示前五层和最后三层，并补入未覆盖注意力类型的代表层（默认包含第 24 层 Reindex，层号从 0 开始）。文本报告不再输出末尾 Notes。

## 配置与来源分离

- `src/models/dsv41flash/config.json`：原始架构快照；保留原字段名。
- `configs/dsv41flash_config.json`：分析场景，包含序列长度、分析精度、Engram/DSpark 开关和训练参数组。
- `architecture_file` 相对于场景 JSON 所在目录解析。`hidden_size/num_layers/vocab_size` 从架构快照导出，不需要重复填写。
- Python API 也可提供内联 `architecture` 字典。场景中如果另填维度且与架构不一致，会报错。
- 修改场景中的 `include_dspark` 为 `true` 可加入 DSpark 参数和独立草稿计算报告；`include_engram=false` 表示整个 Engram 模块从分析范围排除，不能用它代表“只将大表移到 CPU”。

目录与 Pro 保持相同组织方式：模型根目录保留 `__init__.py` 注册入口和 `config.json` 架构配置，核心 Python 实现全部放入 `src/models/dsv41flash/modules/`。

| 文件 | 职责 |
|---|---|
| `model.py` | 模型组装与逐层汇总 |
| `attention.py` | SWA、Full、Reindex、Reuse 以及内部 Compressor/Indexer 计算 |
| `moe.py` | Router、路由专家与共享专家 |
| `mhc.py` | mHC 与子层 RMSNorm |
| `engram.py` | Engram 查表、投影与门控预算 |
| `embedding.py` / `output.py` | 输入 Embedding 与输出 LM Head |
| `dspark.py` | 可选草稿模块的附加投影与输出头 |
| `accounting.py` | 参数张量、精度与训练状态统计 |
| `common.py` | 继承核心 `BaseModule` 的 `FlashModule`，以及共享维度校验和张量工具 |
| `configuration.py` | 场景加载、架构配置解析与校验 |
| `report.py` | Flash 报告展示适配 |

启动命令及场景配置路径保持不变；Python 模型入口仍可使用 `from src.models.dsv41flash import DSV41FlashModel`。安装包包含模型根目录的 `config.json`，省略 `architecture_file` 时使用该配置。

## 模块接口

Flash 与 Pro 使用相同的 `BaseModule` / `ModuleStats` 计算接口。`Attention`、`MoE`、`mHC`、`Engram`、`Embedding`、`OutputLayer` 和三个 DSpark 类均独立实现：

- `compute_params()`：只生成参数明细，返回 `ModuleParamBreakdown` 的列表；不依赖序列长度。
- `compute_flops(seq_len, batch_size=1)`：只计算一次前向的 FLOPs。
- `compute_activations(seq_len, batch_size=1)`：只计算前向激活张量体积（bytes）。
- `analyze(seq_len, batch_size=1)`：复用核心基类汇总三个接口，返回 `ModuleStats`。

`FlashModule` 仅补充参数组、缓存和详细统计元数据。`GroupedParamBreakdown` 在核心参数结构上增加 `group`，保留 dense / expert / engram 的独立精度和训练状态设置。顶层 `DSV41FlashModel` 在初始化时组装模块，分析时调用各模块的 `analyze()`；不再使用一次性计算所有指标的 `build_*` 函数。

```python
from src.core.config_parser import ConfigParser
from src.models.dsv41flash.modules.moe import MoE

config = ConfigParser("configs/dsv41flash_config.json").load()
moe = MoE("MoE layer 0", config, layer_id=0)
params = moe.compute_params()
flops = moe.compute_flops(seq_len=4096, batch_size=1)
activations = moe.compute_activations(seq_len=4096, batch_size=1)
stats = moe.analyze(seq_len=4096, batch_size=1)
```

## 主干与权重归属

零基层号：

| 分类 | 层号 | 数量 | 全局 KV | Indexer |
|---|---|---:|---|---|
| SWA | 0、1 | 2 | 无 | 无 |
| Full | 2、8、14、20 | 4 | 本层生成 | 本层生成 Q、K 和 Top-K |
| Reindex | 24、28、32、36 | 4 | 复用最近来源 | 新 Q/权重，复用 K，重算 Top-K |
| Reuse | 其余主干层 | 30 | 复用最近来源 | 复用 Top-K |

Full/Reindex/Reuse 是统计分类，不是原源码中的 Python 类名。所有 40 层都保留自己的窗口 KV、Q/O 投影。HCA 不存在。只有四个 KV 来源拥有 Compressor 和 Indexer K 参数，不能按 38 个使用全局分支的层重复计数。

| 参数校验范围 | 逻辑参数数量 |
|---|---:|
| 40 层主干及输入输出，排除整个 Engram | 551,566,180,464 |
| 两个 Engram 大表 | 196,613,849,600 |
| 两个 Engram 投影及门控 | 314,654,720 |
| 主干 + 完整 Engram | 748,494,684,784 |
| DSpark 额外参数 | 14,225,362,530 |
| 主干 + Engram + DSpark | 762,720,047,314 |

MoE 按三个 SwiGLU 矩阵计算，不含 TID2EID 表；Router 对全部 384 个专家评分。保留源配置构造出的 `bias_vl`，但不统计视觉塔、aligner、图像边界 embedding。DSpark 与主干共享的输入 embedding 和 LM head 只计一次，Markov embedding/head 是独立参数。

这些是逻辑参数，不含 FP4/FP8 的量化 scale、打包填充和分布式 padding。第一版只接受 BF16/FP16/FP32 分析精度，默认 BF16；不声称其权重容量等于原生量化 checkpoint 或推理驻留容量。

## FLOPs 与动态量口径

- 主报告是 `B×S` 个 token 的主干 teacher-forced 前向估算，乘加计 2 FLOPs；每 token 数值除以 `B×S`。
- Attention 计主要投影 GEMM 和因果有效的 QK/AV 配对，窗口使用 `window_size`，全局配对受压缩比和 `index_topk` 限制。该数值不是具体 kernel 的填充/分块实际执行 FLOPs。
- Indexer 按本地参考代码的全量 QK einsum 计数。参考实现先计算分数，再应用因果/候选 mask，不能声称候选筛选直接减少了这个 GEMM。
- Norm、SiLU、排序、哈希、量化、通信等小算子没有完整计入 FLOPs。未提供反向 FLOPs、耗时、吞吐或硬件利用率模型。
- `activation_memory_bytes` 是逐项列出的前向张量体积之和，包含输入输出重复引用的预算，不是自动微分保存量，也不是同时存活的峰值。JSON 中有每项张量与假设。Indexer 的参考实现中间分数与融合实现可能差别很大。
- `kv_cache_bytes` 是序列结束时的逻辑已填充推理缓存，加上压缩池化状态；只按所有者计数。不是 max-context 预分配容量，不包含 RoPE 表等元数据，也不能直接加进训练激活。
- DSpark 启用后，权重计入总驻留量；一次草稿调用的 FLOPs/张量体积放在 `auxiliary` 中。其 prefix 长度为 `S`，query 长度为 `dspark_block_size`，输入主投影只消费最新一个主 token；不与主干完整序列前向混加。

## 训练参数组与 32 卡场景

| 参数组 | 范围 | 默认优化器状态预算 |
|---|---|---|
| dense | 注意力、Router、共享专家、mHC、输入输出、Engram 投影/门控等 | AdamW 两份 FP32 状态 |
| expert | 路由专家的三个矩阵 | AdamW 两份 FP32 状态 |
| engram | 仅两个 embedding 大表 | momentum 一份 FP32 状态 |

每组均独立配置 `weight_dtype`、`gradient_dtype`、`master_dtype`、`optimizer_dtype`、`optimizer`、`device`、`shards`、`trainable`。参数量由张量形状决定；状态大小由元素数乘各自精度决定，不从压缩权重字节数反推参数量。`master_dtype="none"` 可关闭主权重副本；`trainable=false` 只保留权重，不保留梯度/优化器状态。

优化器选项是容量预算：`adamw` 两份状态，`momentum` 一份，`sgd/none` 零份；不执行真实优化算法。Engram 的 momentum 预算未额外计入更新时临时张量；大表梯度按完整形状预算，未推断稀疏梯度或实际命中行数。

默认配置保留一个未分片模型副本的容量，全部参数组位于 GPU。32 卡示例另设：

- dense 为 32 路分片，可用作 FSDP32 的均衡静态预算；expert 为 32 路专家分片，可用作 EP32 的均衡静态预算。`shards` 是该参数组最终的唯一分片数，不再额外相乘 EP×FSDP。
- Engram 大表及其状态在 CPU，32 路分片；投影、门控和 lookup 输出仍留在 GPU 计算预算中。
- 通信 buffer 1 GB、优化器暂存 2 GB、FSDP cache 2 GB 是显式场景输入，不是架构推导结果。

这是分析场景假设，不是本地推理源码证明的官方训练部署。默认 AdamW 主权重/状态预算与旧 Excel 的优化器统计可能不同，不保证复现旧 PPT 数值。

报告分别列出每卡 GPU/CPU 静态量以及 GPU 静态量加用户 buffer。**没有把未分片的前向张量总量直接加到每卡静态量上并称为训练峰值。** 第一版未实现 TP/PP/CP、激活检查点、重算、专家负载不均、FSDP 预取时序和运行时峰值预测。

## 验证

测试包含独立源码 shape 推导的 golden 参数值、层分类、KV/Indexer 所有者、批量缩放、4K/8K 静态量不变、配置敏感性、DSpark 主权重共享、Engram CPU 放置和冻结、每参数组精度、各级汇总一致性、JSON/CLI、非法配置，以及 Pro 既有数值回归。

该版本不修改 Pro 的 Excel 对齐公式；其既有口径问题仍需独立处理。
