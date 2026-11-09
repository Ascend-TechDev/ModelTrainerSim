# Kimi Next-50T 分析报告

```
===========================================================================================
  Model Analysis Report: Kimi-10T
===========================================================================================

── Configuration ──────────────────────────────────────────────────────────────────────────
  hidden_size         : 7168
  num_layers          : 128
  vocab_size          : 163.84K
  max_seq_len         : 4.10K
  dtype               : bf16

── Parameters ─────────────────────────────────────────────────────────────────────────────
  Total params        : 50.00T
  Embedding params    : 1.17B
  Active params/token : 684.15B
  Weight memory       : 90.94 TB

── Compute (per forward pass) ─────────────────────────────────────────────────────────────
  Seq length          : 4.10K
  Batch size          : 1
  Total FLOPs         : 5.60 PFLOPs
  FLOPs / token       : 1.37 TFLOPs

── Memory ─────────────────────────────────────────────────────────────────────────────────
  Weights             : 90.94 TB
  Activations (fwd)   : 1.54 TB
  KV cache            : 144.00 MB
  Training (weights+grad+opt): 545.67 TB
    - weights                 : 90.94 TB
    - gradients               : 90.94 TB
    - master weights          : 0.00 B
    - optimizer (per_group)   : 363.78 TB

── Per-Module Summary ─────────────────────────────────────────────────────────────────────
  Module          Count           Params            Bytes      FLOPs/token        Act Bytes
  -----------------------------------------------------------------------------------------
  wordembedding       1            1.17B          2.19 GB          0 FLOPs         56.00 MB
  prenorm             1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  postnorm            1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  finalnorm           1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  mla                32            7.43B         13.84 GB     24.52 GFLOPs         39.02 GB
  kda                95           42.16B         78.52 GB     84.28 GFLOPs        147.93 GB
  denseffn            1          726.66M          1.35 GB      1.45 GFLOPs        848.00 MB
  moe               127           49.94T         90.85 TB      1.26 TFLOPs          1.35 TB
  outputlayer         1            1.17B          2.19 GB      2.35 GFLOPs          2.50 GB

── Per-Layer Summary ──────────────────────────────────────────────────────────────────────
  Layer    Attn   FFN   Idx           Params            Bytes            FLOPs
  ----------------------------------------------------------------------------
      0    None Dense False          726.66M          1.35 GB      5.95 TFLOPs
      1     MLA   MoE False          393.49B        732.93 GB     43.63 TFLOPs
      2     MLA   MoE False          393.49B        732.93 GB     43.63 TFLOPs
      3     MLA   MoE False          393.49B        732.93 GB     43.63 TFLOPs
      4     MLA   MoE False          393.49B        732.93 GB     43.63 TFLOPs
    ...
    125     KDA   MoE False          393.70B        733.32 GB     44.13 TFLOPs
    126     KDA   MoE False          393.70B        733.32 GB     44.13 TFLOPs
    127     KDA   MoE False          393.70B        733.32 GB     44.13 TFLOPs

===========================================================================================
```
