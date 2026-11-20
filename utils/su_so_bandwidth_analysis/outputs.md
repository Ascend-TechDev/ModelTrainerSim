# 任务结果表规范

## 文件与字段

| 任务 ID | 文件 | 字段组成 | 列数 |
|---|---|---|---:|
| `base` | [SU_SO场景总表.xlsx](../../SU_SO场景总表.xlsx) | 场景扫描 18 列 + HBM 检查 3 列 | 21 |
| `layer_compute` | [results/单层计算耗时.xlsx](../../results/单层计算耗时.xlsx) | 基础 21 列 + 前向/反向层耗时 | 23 |
| `non_expert_fsdp` | [results/非专家FSDP.xlsx](../../results/非专家FSDP.xlsx) | 原有 31 列 + 16 项通信耗时 + 16 项分项掩盖率 + 24 项整体掩盖率 | 87 |
| `non_expert_fsdp_global_ring` | [results/非专家FSDP_全局Ring.xlsx](../../results/非专家FSDP_全局Ring.xlsx) | 基础 21 列 + 2 项层耗时 + 4 项单卡最大带宽需求 | 27 |
| `non_expert_fsdp_direct_exchange` | [results/非专家FSDP_各Rank直接交互.xlsx](../../results/非专家FSDP_各Rank直接交互.xlsx) | 基础 21 列 + 2 项层耗时 + 4 项单卡均衡带宽需求 | 27 |

EP 已采用独立的维度扫描结果表，见下文；其余任务保留原全量集群场景结构。

非专家 FSDP 的[新增字段](tasks/non_expert_fsdp.md#新增字段)追加在原 31 列末尾；12 个带宽组合按列记录，场景行数不变。基础表和层耗时表不受此扩展影响。

三种通信方式分别存放。全局 Ring 和直接交互两表的末 4 列依次为 `FSDP_fwd_su_bandwidth`、`FSDP_fwd_so_bandwidth`、`FSDP_bwd_su_bandwidth`、`FSDP_bwd_so_bandwidth`，单位 GB/s；通过相同场景 ID 与分层通信表比较。

## E-FSDP 结果文件

三种方式均记录域内/跨域组数、四项单卡最大带宽需求，以及 SU 六档、SO 两档的分项和整体掩盖率；每场景仍占一行。

| 任务 ID | 文件 | 列数 |
|---|---|---:|
| `expert_fsdp` | [results/E-FSDP.xlsx](../../results/E-FSDP.xlsx) | 92 |
| `expert_fsdp_global_ring` | [results/E-FSDP_全局Ring.xlsx](../../results/E-FSDP_全局Ring.xlsx) | 92 |
| `expert_fsdp_direct_exchange` | [results/E-FSDP_各Rank直接交互.xlsx](../../results/E-FSDP_各Rank直接交互.xlsx) | 92 |

三表字段相同：基础 21 列、5 项分组信息、2 项层耗时、8 项流量及带宽、16 项分档通信耗时、16 项分项掩盖率、24 项整体掩盖率。字段顺序、档位和公式见 [E-FSDP 分档及输出规范](tasks/expert_fsdp_results.md)。

## E-DP 结果文件

| 任务 ID | 文件 | 列数 |
|---|---|---:|
| `expert_dp` | [results/E-DP.xlsx](../../results/E-DP.xlsx) | 62 |
| `expert_dp_global_ring` | [results/E-DP_全局Ring.xlsx](../../results/E-DP_全局Ring.xlsx) | 62 |
| `expert_dp_direct_exchange` | [results/E-DP_各Rank直接交互.xlsx](../../results/E-DP_各Rank直接交互.xlsx) | 62 |

每表 62 列：基础 21 列、分组 6 列、层耗时 2 列、梯度分片大小 1 列、反向 SU/SO 流量及带宽 4 列、分档耗时 8 列、分项掩盖率 8 列、整体 All-Reduce 掩盖率 12 列。字段与公式见 [E-DP 输出规范](tasks/expert_dp_results.md)。

## Engram 结果文件

`engram` 独立输出 [results/Engram.xlsx](../../results/Engram.xlsx)，只分析最大 MoE 层对应的一张表，不合并其他 Engram 表的带宽。

共278列：原255列保持名称和顺序，末尾追加前向1层窗口的23列。合计基础21列、层耗时2列、所选表输入5列、窗口10列、流量4列、带宽20列、通信耗时16列、掩盖率200列。前向和反向均采用1/5/10/15/20层完整窗口，不按层号截断。没有对应表的模型填零流量和100%掩盖率；全部模型的共有层耗时保留。字段见 [Engram 输出规范](tasks/engram_results.md)。

## EP 前向维度扫描

方法见 [EP 层内掩盖](tasks/ep.md)，独立结果为 [EP.xlsx](../../results/EP.xlsx)。按 7 个去重中间维×9 个 EP Size×FP8/BF16/DeepEP 完整扫描，汇总 189 行；每种精度 63 行。FP8/BF16 数值相同但分别保留。

共 6 页：全部场景、FP8、BF16、DeepEP、中间维来源、说明。结果页 26 列：场景因子 3 列、远端比例 1 列、两个阶段的有效数据带宽需求 2 列、标称链路需求 2 列、六档 SU 分项掩盖率 12 列，以及追加在末尾的整体掩盖率 6 列。SO 恒为 0，仅在说明中记录。

GEMM 利用率为 95%，All2All 带宽利用率为 70%；掩盖率使用 `min(1,0.7×标称SU带宽/有效需求)`。标称链路需求为有效需求/0.7；两种口径分别记录。固定参数放在说明页及工作簿属性中。

整体字段为 `EP_overall_overlap_ratio_su{L}GBps`，L 按 400/600/800/1000/1200/1600 排列。按通信时间加权：FP8/BF16 为 `(rho_D+rho_C)/2`，DeepEP 为 `(rho_D+2×rho_C)/3`；先分阶段截断，再加权，不借用另一阶段剩余窗口。保存 0～1，显示百分比。

EP 不继承基础 21 列或原场景 ID，不输出 HBM 标记，也不按具体模型的专家数筛选；以“中间维、EP Size、精度组合”关联实际模型，再核查 p 整除 E、p<SU Size、p 整除 SU Size 和容量约束。独立执行 `python3 scripts/generate_ep.py`；原统一入口继续管理十二份集群场景表。

## 最小 SU 带宽配置

[SU最小带宽配置.xlsx](../../results/SU最小带宽配置.xlsx) 独立筛选 HBM 合法配置，仅使用分层 FSDP/E-FSDP/E-DP，并纳入 EP 和 Engram。每模型/精度取各通信项需求的最大值，再在配置中取最小值；EP 按 70% 折算到档位，其余沿用原有口径，不合并争用。

7 页签为模型汇总、最优配置、FP8、BF16、DeepEP、分HBM规格、说明。最优配置 48 行、分 HBM 规格 96 行，明细均 24 列：模型/精度、源场景 ID、SU/N/分组、HBM 占用、最小 SU 需求及匹配档位、瓶颈、各任务 SU 需求和 SO 诊断值。SO 不参与约束。字段和同值择优规则见 [建模方法](tasks/min_su_bandwidth.md)。

## SU/SO 联合带宽配置

[SU_SO联合带宽配置.xlsx](../../results/SU_SO联合带宽配置.xlsx) 共 5 页：模型边界、连续可行边界、档位可行性、最低可行档位、说明。16 模型×3 精度×3 HBM 范围（可选、固定192、固定288）共 144 组，连续边界 198 行；每组全部 12 个档位组合，共 1728 行。

预算组合只有存在同一套 HBM 合法策略同时满足 SU/SO 时才标记可行，明细22列分别记录预算、实际需求、分组和瓶颈。最低可行档位保留所有不被更低预算组合支配的可行组合；当前无档位解的分组保留明确标记。连续边界保留超过既定档位的需求。方法见 [联合配置](tasks/joint_bandwidth.md)。

## 10T/20T 固定 SU、HBM、精度联合配置

[SU_SO联合带宽配置_10T20T_SU576_1152.xlsx](../../results/SU_SO联合带宽配置_10T20T_SU576_1152.xlsx) 对6个模型分别固定 SU Size=576/1152、HBM=192/288 GB、精度=FP8/BF16/DeepEP；每模型12组，共72组，不跨HBM规格择优。每组重新扫描集群规模和全部并行组合。

共7页签：场景最优配置72行、场景因子12行、模型边界72行、连续可行边界75行、档位可行性864行、通信分项明细648行、说明。首表35列：原25列记录固定因子、HBM合法候选数、最低可行预算、完整策略、HBM占用、实际带宽及瓶颈；末10列追加FSDP、E-FSDP、E-DP、EP、Engram各自的SU/SO需求。

每域汇总取其已建模阶段的最大值，全部域再取最大值复原联合需求，不相加。明细页18列，按每个所选策略展开9个阶段，保留模型、固定因子、场景ID、完整并行分组及预算用于关联。EP仅前向且SO为0；Engram采用当前最短的前向1层、反向1层窗口。字段见 [分域带宽](tasks/fixed_su_bandwidth.md#通信域带宽)。

代表策略从同时满足预算的Pareto边界中选取，再以较少卡数等规则择优。当前72组全部有解，SO档位均为100 GB/s，实际SO需求最高89.889992 GB/s。预算与实际需求分列；完整边界保留其他取舍。输入、筛选及择优规则见 [固定三因子方法](tasks/fixed_su_bandwidth.md)。

## 公共约定

以下基础场景及跨表约定适用于十二份集群扫描表；EP 使用上一节的维度场景键及独立输入/方法摘要。

- 前 21 列来自同一套场景扫描和 HBM 检查，字段、数值、场景 ID、行序完全一致；当前保留全部场景及不合法行的计算结果。
- 每份表第一张为“全部场景”，之后按输入顺序每个模型一张；前 5 行为说明，第 6 行为表头，第 7 行起为数据。
- 参数与场景输入来自 CSV/JSON 和公共场景输入文档。任务共享代码中的基础场景与层耗时模块，独立执行时无需读取其他 Excel。
- 单位沿用 [公共方法](common_methods.md)：容量/流量 GB、带宽 GB/s、层耗时和通信耗时 ms。掩盖率保存为 0～1，显示为百分比。

## 版本与核查

工作簿属性记录任务 ID、原始输入文件摘要、解析后模型/场景输入摘要、基础方法摘要和本任务方法摘要。等价 CSV/JSON 的解析后输入摘要相同。

场景 ID 只在同一版基础场景内关联；改变模型或扫描变量后需重新生成全部表。共同基础版本使用 `scenario_basis_sha256` 标识；`modeling_methods_sha256` 按任务依赖的文档分别记录。

每份表独立核查公式、约束、单位和页签一致性。全部任务执行时，另比较各表的基础 21 列、共有层耗时和版本摘要；所有校验通过后，依次替换各自正式文件。比较基线包含完整数值及行序，不受显示舍入影响。

## 增加任务

独立实现计算模块，在 `schema.py` 定义追加字段，在 `task_tables.py` 注册文件、说明及方法依赖，再接入工作流和独立校验。新增任务不扩展其他任务的文件；同一任务新增指标时，在其结果表中追加字段。命令见 [脚本说明](../../scripts/README.md)。
