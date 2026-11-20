# 公共方法与单位

## 单位

| 量 | 内部单位 | 文件/表格单位 |
|---|---|---|
| 参数量 | 个参数 | 输入 `*_params_b` 为 B，1 B=10⁹ 个参数 |
| Engram 每条记录参数量 | 个参数 | 整数 |
| 容量、流量 | byte | GB=10⁹ byte |
| 有效带宽 | byte/s | GB/s；换为 Gb/s 乘 8 |
| 算力 | operations/s | 1 TOPS=10¹² operations/s；乘加计 2 次运算 |
| 时间 | s | 工作簿为 ms，乘 1000 |

工作簿满足 `带宽_GBps × 耗时_ms / 1000 = 流量_GB`。显示舍入不参与计算。

## Collective 流量

采用均衡 Ring 基线，g 为组大小，M 为完整逻辑张量的字节数。

| 操作 | 每卡单向发送量 |
|---|---|
| All-Gather | `(g-1)/g × M`，M 为聚合后大小 |
| Reduce-Scatter | `(g-1)/g × M`，M 为分片前大小 |
| All-Reduce | `2×(g-1)/g × M` |

接收量等于发送量，单向带宽不将两者相加。依据：[NCCL Tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。分层通信按各阶段实际处理的张量大小计算。

## 时间与带宽

```text
T_compute = C / F_eff
T_comm ≈ T_latency + V / (η × BW_physical)
BW_required ≥ V / [η × (W - T_latency)]
```

W 是依赖关系允许的掩盖窗口；有通信时要求 `W>T_latency`。使用有效带宽时不重复乘网络利用率 η。链路负载为经过该链路的字节数之和，共享出口及并发通信需联合核查。

当前 [非专家 FSDP](tasks/non_expert_fsdp.md) 前向采用预取一层、SU/SO 小块流水重叠的掩盖模型，反向采用稳态吞吐模型；暂不计启动延迟和其他通信争用。EP A2A 不扣除层内掩盖收益。
