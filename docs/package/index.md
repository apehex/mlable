# Sources Index

Module-level overview for `mlable/`.

## Top-Level Modules

| Module | Purpose |
| ------ | ------- |
| `-- src/mlable/` | |
| `   -- caching.py` | Cache creation and update helpers used by iterative decoding and attention. |
| `   -- data.py` | Dataset writing/statistics and codepoint decomposition helpers. |
| `   -- masking.py` | Contrast masking utilities for tensor comparisons. |
| `   -- metrics.py` | Grouped accuracy metrics for binary and categorical predictions. |
| `   -- sampling.py` | Top-k/top-p filtering and stochastic sampling functions. |
| `   -- schedules.py` | Numeric schedules, including linear and cosine rate helpers. |
| `   -- shapes.py` | Shape/dimension transformation helpers. |
| `   -- text.py` | UTF/codepoint text utilities and labeling helpers. |
| `   -- utils.py` | Generic utilities (composition, chunking, rotation, numeric helpers). |

## Subpackages

### `src/mlable/blocks`

| Module | Purpose |
| ------ | ------- |
| `-- src/mlable/blocks/` | |
| `   -- attention/` | |
| `      -- generic.py` | attention block implementations and helpers. |
| `      -- transformer.py` | transformer-oriented attention block composition. |
| `   -- convolution/`
| `      -- generic.py` | generic convolutional block patterns. |
| `      -- resnet.py` | ResNet-style convolutional blocks. |
| `      -- unet.py` | U-Net style convolutional blocks. |
| `   -- normalization.py` | adaptive group normalization layer. |
| `   -- shaping.py` | patch extraction and shaping layer utilities. |

### `src/mlable/layers`

| Module | Purpose |
| ------ | ------- |
| `-- src/mlable/layers/` | |
| `   -- embedding.py` | embedding and rotary embedding helper layers/functions. |
| `   -- shaping.py` | Keras layers for divide/merge/swap/move axis transforms. |
| `   -- transformer.py` | feed-forward transformer layer components. |

### `src/mlable/maths`

| Module | Purpose |
| ------ | ------- |
| `-- src/mlable/maths/` | |
| `   -- ops.py` | reduction and grouping operations. |
| `   -- probs.py` | probability utility functions (including log-normal PDF). |

### `src/mlable/models`

| Module | Purpose |
| ------ | ------- |
| `-- src/mlable/models/` | |
| `   -- autoencoder.py` | variational autoencoder model implementation. |
| `   -- diffusion.py` | diffusion model base implementation. |
| `   -- generic.py` | generic model utilities including contrast-oriented models. |

### `src/mlable/shaping`

| Module | Purpose |
| ------ | ------- |
| `-- src/mlable/haping/` | |
| `   -- axes.py` | low-level axis divide/merge/swap/move operations. |
| `   -- hilbert.py` | Hilbert-curve fold/unfold and permutation utilities. |
