# Kimi Next-100T 分析报告

```
===========================================================================================
  Model Analysis Report: Kimi-10T
===========================================================================================

── Configuration ──────────────────────────────────────────────────────────────────────────
  hidden_size         : 7168
  num_layers          : 136
  vocab_size          : 163.84K
  max_seq_len         : 4.10K
  dtype               : bf16

── Parameters ─────────────────────────────────────────────────────────────────────────────
  Total params        : 105.00T
  Embedding params    : 1.17B
  Active params/token : 1.12T
  Weight memory       : 191.00 TB

── Compute (per forward pass) ─────────────────────────────────────────────────────────────
  Seq length          : 4.10K
  Batch size          : 1
  Total FLOPs         : 9.18 PFLOPs
  FLOPs / token       : 2.24 TFLOPs

── Memory ─────────────────────────────────────────────────────────────────────────────────
  Weights             : 191.00 TB
  Activations (fwd)   : 2.25 TB
  KV cache            : 153.00 MB
  Training (weights+grad+opt): 1.12 PB
    - weights                 : 191.00 TB
    - gradients               : 191.00 TB
    - master weights          : 0.00 B
    - optimizer (per_group)   : 764.01 TB

── Per-Module Summary ─────────────────────────────────────────────────────────────────────
  Module          Count           Params            Bytes      FLOPs/token        Act Bytes
  -----------------------------------------------------------------------------------------
  wordembedding       1            1.17B          2.19 GB          0 FLOPs         56.00 MB
  prenorm             1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  postnorm            1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  finalnorm           1            7.17K         14.00 KB          0 FLOPs         56.00 MB
  mla                34            7.89B         14.70 GB     26.06 GFLOPs         41.45 GB
  kda               101           44.82B         83.48 GB     89.60 GFLOPs        157.27 GB
  denseffn            1          726.66M          1.35 GB      1.45 GFLOPs        848.00 MB
  moe               135          104.95T        190.90 TB      2.12 TFLOPs          2.06 TB
  outputlayer         1            1.17B          2.19 GB      2.35 GFLOPs          2.50 GB

── Per-Layer Summary ──────────────────────────────────────────────────────────────────────
  Layer    Attn   FFN   Idx           Params            Bytes            FLOPs
  ----------------------------------------------------------------------------
      0    None Dense False          726.66M          1.35 GB      5.95 TFLOPs
      1     MLA   MoE False          777.62B          1.41 TB     67.50 TFLOPs
      2     MLA   MoE False          777.62B          1.41 TB     67.50 TFLOPs
      3     MLA   MoE False          777.62B          1.41 TB     67.50 TFLOPs
      4     MLA   MoE False          777.62B          1.41 TB     67.50 TFLOPs
    ...
    133     KDA   MoE False          777.83B          1.41 TB     67.99 TFLOPs
    134     KDA   MoE False          777.83B          1.41 TB     67.99 TFLOPs
    135     KDA   MoE False          777.83B          1.41 TB     67.99 TFLOPs

===========================================================================================
```
