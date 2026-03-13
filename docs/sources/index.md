# Sources Index

Module-level overview for `mlable/`.

## Top-Level Modules

| Module | Purpose |
| ------ | ------- |
| `mlable/caching.py` | Cache creation and update helpers used by iterative decoding and attention. |
| `mlable/data.py` | Dataset writing/statistics and codepoint decomposition helpers. |
| `mlable/masking.py` | Contrast masking utilities for tensor comparisons. |
| `mlable/metrics.py` | Grouped accuracy metrics for binary and categorical predictions. |
| `mlable/sampling.py` | Top-k/top-p filtering and stochastic sampling functions. |
| `mlable/schedules.py` | Numeric schedules, including linear and cosine rate helpers. |
| `mlable/shapes.py` | Shape/dimension transformation helpers. |
| `mlable/text.py` | UTF/codepoint text utilities and labeling helpers. |
| `mlable/utils.py` | Generic utilities (composition, chunking, rotation, numeric helpers). |

## Subpackages

### `mlable/blocks`

- `attention/generic.py`: attention block implementations and helpers.
- `attention/transformer.py`: transformer-oriented attention block composition.
- `convolution/generic.py`: generic convolutional block patterns.
- `convolution/resnet.py`: ResNet-style convolutional blocks.
- `convolution/unet.py`: U-Net style convolutional blocks.
- `normalization.py`: adaptive group normalization layer.
- `shaping.py`: patch extraction and shaping layer utilities.

### `mlable/layers`

- `embedding.py`: embedding and rotary embedding helper layers/functions.
- `shaping.py`: Keras layers for divide/merge/swap/move axis transforms.
- `transformer.py`: feed-forward transformer layer components.

### `mlable/maths`

- `ops.py`: reduction and grouping operations.
- `probs.py`: probability utility functions (including log-normal PDF).

### `mlable/models`

- `autoencoder.py`: variational autoencoder model implementation.
- `diffusion.py`: diffusion model base implementation.
- `generic.py`: generic model utilities including contrast-oriented models.

### `mlable/shaping`

- `axes.py`: low-level axis divide/merge/swap/move operations.
- `hilbert.py`: Hilbert-curve fold/unfold and permutation utilities.
