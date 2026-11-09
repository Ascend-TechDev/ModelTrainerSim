# Kimi 10T 宽度-v2 分析报告

```
===========================================================================================
  Model Analysis Report: Kimi-10T
===========================================================================================

── Configuration ──────────────────────────────────────────────────────────────────────────
  hidden_size         : 7168
  num_layers          : 93
  vocab_size          : 163.84K
  max_seq_len         : 4.10K
  dtype               : bf16

── Parameters ─────────────────────────────────────────────────────────────────────────────
  Total params        : 11.92T
  Embedding params    : 1.17B
  Active params/token : 248.80B
  Weight memory       : 21.68 TB

── Compute (per forward pass) ─────────────────────────────────────────────────────────────
  Seq length          : 4.10K
  Batch size          : 1
  Total FLOPs         : 2.05 PFLOPs
  FLOPs / token       : 499.77 GFLOPs

── Memory ─────────────────────────────────────────────────────────────────────────────────
  Weights             : 21.68 TB
  Activations (fwd)   : 670.69 GB
  KV cache            : 108.00 MB
  Training (weights+grad+opt): 130.08 TB
    - weights                 : 21.68 TB
    - gradients               : 21.68 TB
    - master weights          : 0.00 B
    - optimizer (per_group)   : 86.72 TB

── Per-Module Summary ─────────────────────────────────────────────────────────────────────
  Module          Count           Params            Bytes      FLOPs/token        Act Bytes
  -----------------------------------------------------------------------------------------
  wordembedding       1            1.17B          2.19 GB          0 FLOPs         56.00 MB
  prenorm             1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  postnorm            1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  finalnorm           1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  mla                24            5.57B         10.38 GB     18.39 GFLOPs         29.26 GB
  kda                69           30.62B         57.03 GB     61.21 GFLOPs        107.44 GB
  denseffn            1          726.66M          1.35 GB      1.45 GFLOPs        848.00 MB
  moe                92           11.88T         21.61 TB    416.36 GFLOPs        530.44 GB
  outputlayer         1            1.17B          2.19 GB      2.35 GFLOPs          2.50 GB

── Per-Layer Summary ──────────────────────────────────────────────────────────────────────
  Layer    Attn   FFN   Idx           Params            Bytes            FLOPs
  ----------------------------------------------------------------------------
      0     MLA Dense False          958.86M          1.79 GB      9.09 TFLOPs
      1     MLA   MoE False          129.35B        240.92 GB     21.68 TFLOPs
      2     MLA   MoE False          129.35B        240.92 GB     21.68 TFLOPs
      3     MLA   MoE False          129.35B        240.92 GB     21.68 TFLOPs
      4     MLA   MoE False          129.35B        240.92 GB     21.68 TFLOPs
    ...
     90     KDA   MoE False          129.56B        241.32 GB     22.17 TFLOPs
     91     KDA   MoE False          129.56B        241.32 GB     22.17 TFLOPs
     92     KDA   MoE False          129.56B        241.32 GB     22.17 TFLOPs

===========================================================================================
```
