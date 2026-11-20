# Engram 输出规范

独立输出 [results/Engram.xlsx](../../../results/Engram.xlsx)，建模公式见 [Engram 按需查表通信](engram.md)。

## 组织与单位

保留全部 16 个模型及原场景行序，第一页为全部场景，随后每模型一页。只统计所选最大 MoE 层的 Engram 表；层号使用基础列 `所选MoE层ID`，来自输入的 `layer_id_zero_based`。不合法场景仍计算，后续分析筛选 HBM 合法标记为 1。

参数量 B=10^9 参数；每条记录的参数数按个参数、记录条数按条；流量 GB=10^9 byte；带宽 GB/s；窗口及耗时 ms；掩盖率保存为 0～1，显示百分比。

## 字段顺序

前向和反向均扫描1、5、10、15、20层。保留原255列的名称和顺序，新增前向1层的23列统一追加在末尾；各指标的总数如下。

| 字段 | 内容 | 列数 |
|---|---|---:|
| 原基础场景字段 | 场景、模型属性和 HBM 判定 | 21 |
| `layer_fwd_time`、`layer_bwd_time` | 已有整层耗时 | 2 |
| `Engram_table_params_b`、`Engram_table_rows`、`Engram_record_params` | 所选一张表的参数量、记录数、记录宽度 | 3 |
| `Engram_lookups_per_token`、`Engram_lookup_count` | λ、Q=t×λ | 2 |
| `Engram_{phase}_window_{d}layers` | 完整 d 层计算窗口 | 10 |
| `Engram_{phase}_{X}_vol` | 前向索引+记录，反向索引+梯度 | 4 |
| `Engram_fwd_{X}_bandwidth_prefetch_{d}layers` | 提前 d 层预取的带宽需求，5×2 | 10 |
| `Engram_bwd_{X}_bandwidth_window_{d}layers` | d 层反向预算的带宽需求，5×2 | 10 |
| `Engram_{phase}_{X}_comm_time_{B}GBps` | 对应链路档位的通信耗时，不按窗口重复 | 16 |
| 下述窗口分项掩盖率 | 每窗口 SU 六档、SO 两档，10×8 | 80 |
| 下述窗口整体掩盖率 | 每窗口 12 个 SU/SO 组合，10×12 | 120 |
| **合计** | 档位和深度按列展开，不增加场景行数 | **278** |

原255列中，phase先fwd后bwd，X先su后so；前向深度5/10/15/20，反向1/5/10/15/20。通信耗时按前向SU六档、前向SO两档、反向SU六档、反向SO两档排列；每窗口掩盖率先8项分项，再12项整体。

末尾新增23列依次为：`Engram_fwd_window_1layers`，SU/SO两项`bandwidth_prefetch_1layers`，SU六档及SO两档的8项分项掩盖率，12项整体掩盖率。命名沿用以下规则：

```text
Engram_fwd_{X}_overlap_ratio_prefetch_{d}layers_{B}GBps
Engram_bwd_{X}_overlap_ratio_window_{d}layers_{B}GBps
Engram_fwd_lookup_overlap_ratio_prefetch_{d}layers_su{SU}GBps_so{SO}GBps
Engram_bwd_gradient_overlap_ratio_window_{d}layers_su{SU}GBps_so{SO}GBps
```

SU 档位为 400、600、800、1000、1200、1600；SO 为 100、200 GB/s。整体组合按 SU 升序、每档下 SO 升序排列，取两项掩盖率的较小值。

## 边界与版本

- 窗口 d×T 不按目标层号截断；不合并其他 Engram 表的带宽。
- 无对应表时，除已有层耗时和掩盖率外，Engram 字段均为 0；掩盖率为 1。
- 有表但 k=1 时，仅 SO 流量/带宽/通信耗时为 0，SO 掩盖率为 1，整体由 SU 决定。
- 每卡流量为均衡条件下的单向负载，不将发送与接收相加。
- 工作簿记录输入/方法摘要，以及所选表输入、索引字节数、十组完整窗口、访问方式和链路档位。公共基础字段和层耗时与已有表一致。

字段定义见 [engram_schema.py](../../../scripts/su_so_modeling/engram_schema.py)。独立调用计算或导出 API 时，自定义 CSV/JSON 应先经 `read_engram_inputs(path)` 解析为 `task_inputs` 并传递；命令行 `--input` 自动完成此步骤。
