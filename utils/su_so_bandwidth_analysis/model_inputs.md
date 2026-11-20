# 模型输入与参数口径

## 来源

读取根目录 [max_layer_parameters.csv](../../max_layer_parameters.csv) 或 [max_layer_parameters.json](../../max_layer_parameters.json)，任选一份。当前覆盖 16 个模型、26 个字段；JSON 结构为 `{"models": [...]}`，CSV 为 UTF-8 BOM。

`source_report` 仅用于追溯。分析任务不从报告补入未声明的字段。输入生成及配置对齐命令见 [脚本说明](../../scripts/README.md)。

## 统计口径

- 模型级参数覆盖所有层、Embedding、输出头及层外参数；模型级路由专家参数按所有 MoE 层累加。
- 最大层只从实际 MoE 层选取，比较包含 Engram 表的层总参数；并列选最小零起始层 ID，不拼接不同层的最大值。
- 非专家字段含共享专家、Router、注意力、Norm、Latent/Engram 投影和门控，排除路由专家及单列的 Engram 查表权重。
- E、Top-K 为每个 MoE 层的值，不按层数累加；当前每个模型的各 MoE 层一致。
- 所有 `*_params_b` 为 B=10⁹ 个参数，精确到单个参数；`*_record_params` 为整数个参数。
- Engram 按层 ID 分列，当前为 layer_1、layer_14；无表时对应三项为 0。新增 Engram 层需扩展字段，同层多表需先明确统计方式。
- `hidden_size` 是残差流维度，`moe_expert_hidden_size` 是专家 FFN 中间维度。EP 实际传输维度取决于分发位置，不能直接用 FFN 中间维度代替。

## 字段

| 字段 | 含义 / 单位 |
|---|---|
| `model_id` | 稳定模型 ID |
| `model` | 模型名称 |
| `source_report` | 完整报告路径，相对于工作区根目录 |
| `model_total_params_b` | 完整模型总参数，B，包含 Dense 层及层外参数 |
| `model_active_params_b` | 完整模型每 token 激活参数总量，B |
| `model_non_expert_params_b` | 完整模型非专家参数总量，B；包括共享专家、Router，剔除 Engram 表 |
| `model_routed_expert_total_params_b` | 所有 MoE 层的路由专家参数之和，B |
| `model_routed_expert_count` | 每个 MoE 层的路由专家数量 E，个；不按层数相加 |
| `model_active_routed_expert_count` | 每个 MoE 层每 token 激活专家数 Top-K，个 |
| `hidden_size` | 模型 hidden size，整数维度 |
| `moe_expert_hidden_size` | 专家 FFN 中间维度，即 Excel 的 MoE Hidden Dim |
| `model_engram_layer_1_params_b` | 第 1 层整张 Engram 表参数量，B |
| `model_engram_layer_1_rows` | 第 1 层 Engram 表记录条数 |
| `model_engram_layer_1_record_params` | 第 1 层 Engram 每条记录参数量，个 |
| `model_engram_layer_14_params_b` | 第 14 层整张 Engram 表参数量，B |
| `model_engram_layer_14_rows` | 第 14 层 Engram 表记录条数 |
| `model_engram_layer_14_record_params` | 第 14 层 Engram 每条记录参数量，个 |
| `layer_id_zero_based` | 参数量最大的 MoE 层 ID，0 起算 |
| `layer_total_params_b` | 所选 MoE 层完整参数量，B，包含本层 Engram 表 |
| `layer_non_expert_params_b` | 所选层非专家参数，B；含共享专家、Router，剔除 Engram 表 |
| `layer_routed_expert_total_params_b` | 所选层全部路由专家参数，B |
| `layer_routed_expert_count` | 所选层路由专家数量，个 |
| `layer_active_routed_expert_count` | 所选层每 token 激活路由专家数量，个 |
| `layer_engram_table_params_b` | 所选层整张 Engram 表参数量，B |
| `layer_engram_table_rows` | 所选层 Engram 表记录条数 |
| `layer_engram_record_params` | 所选层 Engram 每条记录参数量，个 |

## 派生量与守恒

以下 P 均为个参数：

```text
P_model_non = 10^9 × (model_non_expert_params_b + Σ model_engram_layer_<id>_params_b)
P_model_route = 10^9 × model_routed_expert_total_params_b
P_model_total = P_model_non + P_model_route

P_layer_non_comm = 10^9 × layer_non_expert_params_b
P_layer_route = 10^9 × layer_routed_expert_total_params_b
P_layer_table = 10^9 × layer_engram_table_params_b
P_layer_total = P_layer_non_comm + P_layer_route + P_layer_table

E = layer_routed_expert_count
K = layer_active_routed_expert_count
P_expert = P_layer_route / E
P_layer_active_compute = P_layer_non_comm + K × P_expert
P_layer_engram_active = n_lookup × layer_engram_record_params
P_layer_active = P_layer_active_compute + P_layer_engram_active
```

`n_lookup` 为每 token 查询记录数，当前输入未提供。查询记录计入访问/激活参数统计，但不作为矩阵权重计入层计算量。容量包含整张表，非专家 FSDP 通信排除查表权重。

校验模型和层参数分类守恒，以及 `表参数量_B×10^9=记录条数×每条记录参数量`。CSV/JSON 数值相同；所有表和专家参数使用完整逻辑量，不预先应用并行分片。
