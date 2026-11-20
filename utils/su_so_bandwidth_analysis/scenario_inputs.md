# 公共场景输入

本文件是脚本读取场景参数的唯一来源。模型参数来自 [模型输入](model_inputs.md)；单位见 [公共方法](common_methods.md)。

## 训练精度

Muon 优化器只计一份 FP32 状态。

| 项目 | 符号 | FP8 训练 | BF16 训练 |
|---|---|---|---|
| 权重参数 | `b_weight` | FP8，1 byte/参数 | BF16，2 byte/参数 |
| 梯度存储 | `b_gradient_storage` | BF16，2 byte/参数 | BF16，2 byte/参数 |
| 优化器状态 | `b_optimizer` | FP32，4 byte/参数 | FP32，4 byte/参数 |
| 主权重 | `b_master` | FP32，4 byte/参数 | FP32，4 byte/参数 |
| 参数梯度通信 | `b_gradient_comm` | FP32，4 byte/元素 | FP32，4 byte/元素 |
| 基础训练状态合计 | `b_state` | 11 byte/参数 | 12 byte/参数 |

HBM 检查仅计四类训练状态；激活、通信缓冲区、量化元数据、临时工作区不计入。

## 算力与负载

所有计算使用对应精度的单卡算力，`F_eff = F_peak × u_compute`。

| 场景 | 计算精度 | 峰值算力 `F_peak` | 利用率 `u_compute` | 有效算力 `F_eff` |
|---|---|---:|---:|---:|
| FP8 | FP8 | 3538 TOPS | 40% | 1415.2 TOPS |
| BF16 | BF16 | 1769 TOPS | 40% | 707.6 TOPS |

| 项目 | 符号 | 固定值 |
|---|---|---:|
| 单卡 micro-batch | `micro_batch_size` | 1 |
| 序列长度 | `seq_len` | 2k = 2048 token |
| 单卡原始 token 数 | `t_card = micro_batch_size × seq_len` | 2048 |
| 反向/前向计算耗时比 | `T_layer_bwd / T_layer_fwd` | 2 |

序列长度的后缀 k 表示 1024，与下文 SU 域数量 k 无关。计算利用率不重复折减；原始 token 数不按 Top-K 展开。

## 拓扑

| 变量 | 含义 | 范围或约束 |
|---|---|---|
| `S = SU Size` | 单 SU 卡数 | `[128, 256, 288, 512, 576, 1024, 1152, 2048, 4096, 8192, 12288]` |
| `k` | SU 域数量 | `[1, 2, 4, 8, 16, 32, 64, 128]` |
| `N = SO Size` | 集群总卡数 | `N = S × k`，`N < 200000` |
| `H_card` | 单卡 HBM | `[192, 288] GB` |
| `hbm_limit_ratio` | 训练状态容量上限比例 | 0.95 |

`k=1` 时无跨域 SO 通信；同总卡数、不同 SU 划分分别保留。

## 并行策略与分组

仅使用 FSDP、EP、E-FSDP、E-DP。定义 `p=EP Size`、`F=E-FSDP Size`、`R=E-DP Size`：

```text
FSDP Size = N
N = p × F × R
EP Size ∈ [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
```

- 非专家 FSDP 覆盖 N 卡，存储范围包括共享专家、Router 和 Engram。
- EP 按专家数量分配，每个 rank 负责 `E/p` 个等规模路由专家。
- E-FSDP 对不同 EP 域中相同 EP rank 的专家参数分片。
- E-DP 复制完整 MoE；每个副本占 `p×F` 卡，共 R 份。

扫描要求 `p < S`、`S % p = 0`、`E % p = 0`。EP 组完整位于 SU 内；F 遍历 `N/p` 的所有正整数约数，`R=N/(p×F)`。

用 `(r,f,e)` 表示逻辑 rank：

| 通信组 | 固定坐标 | 变化坐标 | 大小 |
|---|---|---|---:|
| EP | r、f | e | p |
| E-FSDP | r、e | f | F |
| E-DP | f、e | r | R |
| 非专家 FSDP | 无 | 全部 | N |

MoE 计算基线采用完全均衡路由，无丢 token、填充或去重。物理布局按 EP → E-FSDP → E-DP 的优先级，域内完整分组及剩余成员的跨域分组规则见 [E-FSDP](tasks/expert_fsdp.md#物理分组先组成域内组再处理剩余成员)。
