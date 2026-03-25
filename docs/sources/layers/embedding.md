# `mlable.layers.embedding`

Embedding and positional encoding layers.

## `TokunEmbedding`

Embeds token components independently, then merges component and embedding axes.

```python
import tensorflow as tf
import mlable.layers.embedding

x = tf.random.uniform((2, 32, 4), minval=0, maxval=256, dtype=tf.int32)
layer = mlable.layers.embedding.TokunEmbedding(input_dim=256, output_dim=128)
y = layer(x)

list(y.shape)
# [2, 32, 512]
```

Input shape convention:

- `... x sequence x token_parts`

Output shape:

- `... x sequence x (token_parts * output_dim)`

## `RotaryPositionalEmbedding`

Applies RoPE on configurable sequence and feature axes.

```python
import tensorflow as tf
import mlable.layers.embedding

x = tf.ones(shape=(1, 4, 6), dtype=tf.float32)
layer = mlable.layers.embedding.RotaryPositionalEmbedding(
    sequence_axis=1,
    feature_axis=-1,
    max_wavelength=10_000,
    scaling_factor=1.0,
)
y = layer(inputs=x, offset=0)
```

Arguments:

- `sequence_axis`: axis used as positional dimension.
- `feature_axis`: axis used as feature dimension.
- `max_wavelength`: RoPE wavelength parameter.
- `scaling_factor`: position scaling factor.
- `offset` (call-time): useful for iterative decoding.

## Related utilities

This module also provides `CosineEmbedding`, `SinePositionalEmbedding`, and `PositionalEmbedding`.

Behavior and shape invariants are covered in `tests/layers/test_embedding.py`.
