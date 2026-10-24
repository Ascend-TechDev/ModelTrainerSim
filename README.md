# ModelTrainerSimulator

LLM 架构参数建模与显存/算力分析工具包。

已新增 **DeepSeek V4.1 Flash**：按源码拆分 40 层 SWA/Full/Reindex/Reuse、MoE、Engram、输入输出与可选 DSpark，支持按参数组配置训练状态、设备放置和分片，容量统一按十进制 GB 显示。使用方式、数据来源与统计边界见 [V4.1 Flash 分析说明](docs/DSV41_FLASH.md)。

```bash
python -m src.main --config configs/dsv41flash_config.json
python -m src.main --config configs/dsv41flash_32gpu_config.json --seq-len 8192
python -m unittest tests.test_dsv41flash -v
```

Flash 的前向张量体积预算与训练峰值不同；Pro 保留原有 Excel 对齐计算口径。

Flash 的计算模块与 Pro 共用 `BaseModule` / `ModuleStats` 接口，分别实现 `compute_params()`、`compute_flops()`、`compute_activations()`，由顶层模型统一调用 `analyze()` 汇总。

支持通过 `config.json` 配置模型架构参数，自动计算各模块（CSA / HCA / mHC / Indexer / MoE）的：
- **参数量** (Parameter count)
- **显存占用** (Weight / Activation / KV-cache memory)
- **运算量** (FLOPs)
- **训练显存估算** (Gradients + Optimizer states)

## 项目结构

```
ModelTrainerSimluator/
├── src/
│   ├── main.py                      # CLI 统一入口
│   ├── core/                        # 核心框架
│   │   ├── base_module.py           # BaseModule 抽象基类
│   │   ├── config_parser.py         # JSON 配置解析器
│   │   ├── model_registry.py        # 模型注册表（支持扩展）
│   │   ├── dtype.py                 # 数据类型定义 (fp32/bf16/fp8/int4...)
│   │   ├── formatter.py             # 数字/字节格式化工具
│   │   └── reporter.py              # 报告渲染器
│   └── models/                      # 各模型家族
│       ├── dsv4pro/                 # DeepSeek V4 Pro
│       │   └── modules/
│       │       ├── csa.py           # Compressed Sparse Attention
│       │       ├── hca.py           # Hierarchical Compressed Attention
│       │       ├── mhc.py           # mini Hierarchical Compression
│       │       ├── indexer.py       # Sparse Block Indexer
│       │       ├── moe.py           # Mixture of Experts
│       │       └── model.py         # DSV4ProModel 顶层组装
│       ├── dsv41flash/              # DeepSeek V4.1 Flash
│       │   ├── __init__.py          # 模型注册入口
│       │   ├── config.json          # 原始架构配置
│       │   └── modules/            # 核心实现，与 Pro 同级组织
│       │       ├── model.py        # DSV41FlashModel 顶层组装
│       │       ├── attention.py    # SWA / Full / Reindex / Reuse
│       │       ├── moe.py          # Router 与专家
│       │       ├── mhc.py          # mHC 与 RMSNorm
│       │       ├── engram.py       # Engram
│       │       ├── embedding.py    # 输入 Embedding
│       │       ├── output.py       # 输出 LM Head
│       │       ├── dspark.py       # DSpark 附加模块
│       │       ├── accounting.py   # 参数与训练状态统计
│       │       ├── common.py       # 模块公共工具
│       │       ├── configuration.py # 配置加载与校验
│       │       └── report.py       # 报告展示
│       ├── kimi/                    # Kimi 10T (MLA + KDA + Latent MoE)
│       ├── glm/                     # GLM-5.2 (MoE + DSA + Indexer)
│       ├── qwen/                    # (placeholder) TODO
│       └── hunyuan/                 # Hunyuan Hy4 (10T: iHC + Gated MLA + 2048-expert MoE)
├── configs/
│   ├── dsv4pro_config.json          # DSV4Pro 配置文件
│   ├── dsv41flash_config.json       # Flash 默认分析场景
│   └── dsv41flash_32gpu_config.json # Flash 32 卡分析场景
├── examples/
│   └── quick_start.py               # 示例脚本
├── tests/
│   ├── test_dsv4pro.py              # Pro 单元测试
│   └── test_dsv41flash.py           # Flash 单元测试
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 快速开始

### CLI 使用

```bash
# 分析 DSV4Pro 模型
python -m src.main --config configs/dsv4pro_config.json

# 指定序列长度和 batch size
python -m src.main --config configs/dsv4pro_config.json --seq-len 8192 --batch-size 4

# 输出 JSON 格式
python -m src.main --config configs/dsv4pro_config.json --format json --output report.json

# 列出所有已注册的模型家族
python -m src.main --list-models

# 分析 Hunyuan Hy4-preview 10T（iHC 四残差流 + Gated MLA + 2048 专家 MoE）
python -m src.main --config configs/hy4_10t_config.json

# 分析 Kimi 10T（深度-v2: 137 层 MLA+KDA 混合注意力 + Latent MoE；宽度-v2: 93 层）
python -m src.main --config configs/kimi_10t_deep_v2_config.json
python -m src.main --config configs/kimi_10t_wide_v2_config.json
```

### Python API 使用

```python
from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry
from src.core.reporter import render_report

# 1. 加载配置
config = ConfigParser("configs/dsv4pro_config.json").load()

# 2. 构建模型
model = ModelRegistry.build(config["model"], config)

# 3. 分析（参数量、显存、FLOPs）
report = model.analyze(seq_len=4096, batch_size=1)

# 4. 查看报告
print(render_report(report))

# 或直接访问数据
print(f"Total params: {report['model_params']['total_params']:,}")
print(f"Weight memory: {report['model_params']['total_weight_bytes'] / 1e9:.2f} GB")
print(f"FLOPs/token: {report['compute']['flops_per_token']:.2e}")
```

## DSV4Pro 官网架构更新

DSV4Pro 引入了以下核心创新模块：

| 模块 | 全称 | 说明 |
|------|------|------|
| **CSA** | Compressed Sparse Attention | 低秩压缩 + 稀疏 top-k 注意力，用于早期层 |
| **HCA** | Hierarchical Compressed Attention | 分层（block级 + token级）压缩注意力，用于长上下文层 |
| **mHC** | mini Hierarchical Compression | 轻量级压缩注意力 + GQA，用于后期层 |
| **Indexer** | Sparse Block Indexer | 路由模块，决定每个 query 访问哪些 KV 块 |
| **MoE** | Mixture of Experts | 细粒度专家网络 + 共享专家 + GLSE 路由 |

### 配置示例 (`config.json`)

```json
{
  "model": "DSV4Pro",
  "hidden_size": 7168,
  "num_layers": 61,
  "vocab_size": 129280,
  "max_seq_len": 131072,
  "dtype": "bf16",
  "layers": {
    "first_k_dense_replace": 3,
    "csa_layers": 3,
    "hca_layers": 16,
    "attention": {
      "kv_lora_rank": 512,
      "num_heads": 128,
      "head_dim": 192
    },
    "moe": {
      "moe_intermediate_size": 2048,
      "num_experts": 256,
      "num_activated_experts": 8,
      "num_shared_experts": 1
    }
  }
}
```

## 扩展新模型家族

添加新模型（如 Kimi、GLM、Qwen、Hunyuan）只需三步：

1. **创建模块类**：继承 `BaseModule`，实现 `compute_params()` 和 `compute_flops()`：

```python
# src/models/qwen/modules/attention.py
from ....core.base_module import BaseModule, ModuleParamBreakdown

class QwenAttention(BaseModule):
    module_type = "QwenAttention"

    def compute_params(self):
        # 返回参数分解
        ...

    def compute_flops(self, seq_len, batch_size=1):
        # 返回 FLOPs
        ...
```

2. **创建模型类**：组装各模块，参考 `src/models/dsv4pro/modules/model.py`

3. **注册模型**：在 `src/models/<family>/__init__.py` 中注册：

```python
from ...core.model_registry import ModelRegistry

ModelRegistry.register("Qwen")(QwenModel)
```

4. **更新 `model_registry.py`** 的自动发现逻辑

### BaseModule 接口

| 方法 | 用途 | 必须 |
|------|------|------|
| `compute_params()` | 返回 `list[ModuleParamBreakdown]`，每个子权重张量的形状和数量 | ✅ |
| `compute_flops(seq_len, batch_size)` | 返回该模块前向一次的 FLOPs | ✅ |
| `compute_activations(seq_len, batch_size)` | 返回前向激活显存（bytes），默认 0 | ❌ |
| `analyze(seq_len, batch_size)` | 便捷方法，返回完整 `ModuleStats` | ✅ (已实现) |

## 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
