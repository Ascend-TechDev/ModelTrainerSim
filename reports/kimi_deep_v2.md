# Kimi 10T 深度-v2 分析报告

```
===========================================================================================
  Model Analysis Report: Kimi-10T
===========================================================================================

── Configuration ──────────────────────────────────────────────────────────────────────────
  hidden_size         : 7168
  num_layers          : 137
  vocab_size          : 163.84K
  max_seq_len         : 16.38K
  dtype               : bf16

── Parameters ─────────────────────────────────────────────────────────────────────────────
  Total params        : 9.28T
  Embedding params    : 1.17B
  Active params/token : 155.06B
  Weight memory       : 16.89 TB

── Compute (per forward pass) ─────────────────────────────────────────────────────────────
  Seq length          : 16.38K
  Batch size          : 1
  Total FLOPs         : 5.65 PFLOPs
  FLOPs / token       : 344.81 GFLOPs

── Memory ─────────────────────────────────────────────────────────────────────────────────
  Weights             : 16.89 TB
  Activations (fwd)   : 1.96 TB
  KV cache            : 612.00 MB
  Training (weights+grad+opt): 101.32 TB
    - weights                 : 16.89 TB
    - gradients               : 16.89 TB
    - master weights          : 0.00 B
    - optimizer (per_group)   : 67.55 TB

── Per-Module Summary ─────────────────────────────────────────────────────────────────────
  Module          Count           Params            Bytes      FLOPs/token        Act Bytes
  -----------------------------------------------------------------------------------------
  wordembedding       1            1.17B          2.19 GB          0 FLOPs        224.00 MB
  prenorm             1            7.17K         14.00 KB          0 FLOPs        224.00 MB
  postnorm            1            7.17K         14.00 KB          0 FLOPs        224.00 MB
  finalnorm           1            7.17K         14.00 KB          0 FLOPs        224.00 MB
  mla                34            7.89B         14.70 GB     56.86 GFLOPs        165.82 GB
  kda               102           45.26B         84.31 GB     90.49 GFLOPs        635.31 GB
  denseffn            1          726.66M          1.35 GB      1.45 GFLOPs          3.31 GB
  moe               136            9.23T         16.78 TB    193.66 GFLOPs          1.17 TB
  outputlayer         1            1.17B          2.19 GB      2.35 GFLOPs         10.00 GB

── Per-Layer Summary ──────────────────────────────────────────────────────────────────────
  Layer    Attn   FFN   Idx           Params            Bytes            FLOPs
  ----------------------------------------------------------------------------
      0    None Dense False          726.66M          1.35 GB     23.81 TFLOPs
      1     MLA   MoE False           68.08B        126.80 GB     50.73 TFLOPs
      2     MLA   MoE False           68.08B        126.80 GB     50.73 TFLOPs
      3     MLA   MoE False           68.08B        126.80 GB     50.73 TFLOPs
      4     MLA   MoE False           68.08B        126.80 GB     50.73 TFLOPs
    ...
    134     KDA   MoE False           68.29B        127.20 GB     37.87 TFLOPs
    135     KDA   MoE False           68.29B        127.20 GB     37.87 TFLOPs
    136     KDA   MoE False           68.29B        127.20 GB     37.87 TFLOPs

===========================================================================================
```
