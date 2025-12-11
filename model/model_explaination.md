## Baseline model: `repvit_m0_9`

The four stages are separated by blocks with stride `s = 2`. Unless otherwise specified, the input resolution is 224x224.

- **Stem (not counted as a stage)**  
  Two 3x3 convolutions with stride 2, each followed by BN and activation. These reduce the spatial size from 224x224 -> 112x112 -> 56x56 and channels from 3 -> 24 -> 48. The output is a 56x56x48 feature map that feeds Stage 1.

- **Stage 1**  
  - **Resolution / width / depth:** 56x56, C = 48, 3 blocks (rows 1–3 in the config).  
  - **Structure:** all blocks use stride 1. The token mixer is **RepVGGDW** (DW 3x3 + DW 1x1 + identity; three branches during training, re-parameterized to a single DW conv at deployment). The channel mixer is a 1x1 -> activation -> 1x1 sequence with expansion ratio `t = 2` and a residual connection.  
  - **SE and activation:** SE pattern `[1, 0, 0]`; `use_hs = 0` (standard GELU / ReLU).

- **Stage 2**  
  - **Resolution / width / depth:** 28x28, C = 96, 4 blocks (rows 4–7).  
  - **Structure:** starts with a downsampling block (DW kxk, stride 2 -> 1x1 conv to change channels), followed by 3 stride-1 blocks, all with `t = 2`.  
  - **SE and activation:** SE pattern `0 | 1, 0, 0` (no SE on the downsampling block); `use_hs = 0`.

- **Stage 3 (main compute stage)**  
  - **Resolution / width / depth:** 14x14, C = 192, 16 blocks (rows 8–23).  
  - **Structure:** the first block downsamples (stride 2); the remaining 15 use stride 1. Token mixer is the same as above, with `t = 2`.  
  - **SE and activation:** SE is roughly alternating (the first stride-2 block has SE disabled, followed by 1,0,1,0,… on subsequent blocks). `use_hs = 1` (configured for an h-swish–style activation, depending on the implementation).

- **Stage 4**  
  - **Resolution / width / depth:** 7x7, C = 384, 3 blocks (rows 24–26).  
  - **Structure:** first block downsamples (stride 2), followed by 2 stride-1 blocks; `t = 2`.  
  - **SE and activation:** SE pattern `0 | 1, 0`; `use_hs = 1`.

- **Head**  
  Global average pooling (GAP) followed by a BN + linear classifier. For 100-way classification, the head is 384 -> 100. At deployment, BN and the linear layer can be fused.

**Additional notes**

- All four stages use kernel size `k = 3` in the baseline; the expansion ratio is fixed at `t = 2`.  
- Blocks with stride 2 are responsible for both downsampling and channel change, and SE is typically disabled on these blocks.  
- Depth allocation is `3 / 4 / 16 / 3` across the four stages; Stage 3 at low resolution hosts most of the compute to improve accuracy and semantic capacity.

---

## Depth variants

- **`repvit_m0_9_d22142`**  
  RepViT-M0.9 with stage depths changed to `[2, 2, 14, 2]`.

- **`repvit_m0_9_d22182`**  
  Stage depths `[2, 2, 18, 2]`: additional computation is placed in **Stage 3 (14x14, C = 192)**.  
  *Goal:* deepen the network at lower resolution to gain accuracy while keeping latency under control.

- **`repvit_m0_9_d24122`**  
  Stage depths `[2, 4, 12, 2]`: Stage 2 (28x28, C = 96) is deepened, while Stage 3 is slightly reduced.  
  *Goal:* test the effect of mid-scale texture/edge representation on accuracy and reduce the cost of the high-channel later stages.

---

## Width variants

- **`repvit_m0_9_w088`**  
  Width −12%: `48, 96, 192, 384` -> `42, 84, 168, 336` (aligned to multiples of 8).

- **`repvit_m0_9_w112`**  
  Width +12%: `48, 96, 192, 384` -> approximately `54, 108, 216, 432` (aligned to multiples of 8).

- **`repvit_m0_9_w125`**  
  Width +25%: `48, 96, 192, 384` -> `60, 120, 240, 480` (aligned to multiples of 8).

---

## Kernel-size variants

- **`repvit_m0_9_k5555`**  
  All stages use kernel size 5x5.

- **`repvit_m0_9_k3355`**  
  Stages 1–2 use 3x3; stages 3–4 use 5x5.

- **`repvit_m0_9_k3353`**  
  Only Stage 3 uses 5x5; all other stages keep 3x3.

---

## SE-pattern variants

- **`repvit_m0_9_se_none`**  
  All SE layers disabled.

- **`repvit_m0_9_se_alt`**  
  SE alternates on/off for stride-1 blocks within each stage (1, 0, 1, 0, …); all stride-2 downsampling blocks keep SE disabled.

- **`repvit_m0_9_se_all`**  
  All stride-1 blocks use SE; stride-2 downsampling blocks are still kept without SE, following the common design choice for boundary blocks.

