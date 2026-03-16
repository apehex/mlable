# `mlable.layers.transformer`

Transformer-oriented layers.

## `FeedForwardNetwork`

Dense -> dropout -> dense projection with configurable hidden size.

```python
import tensorflow as tf
import mlable.layers.transformer

x = tf.ones((1, 2), dtype=tf.float32)
layer = mlable.layers.transformer.FeedForwardNetwork(
    hidden_dim=4,
    use_bias=False,
    activation='gelu',
)
y = layer(x, training=False)
```

Arguments:

- `hidden_dim`: intermediate dense width.
- `use_bias`: whether dense layers include bias.
- `dropout_rate`: dropout applied after hidden projection.
- `activation`: hidden activation (default `gelu`).

## `CachedMultiHeadAttention`

Subclass of `tf.keras.layers.MultiHeadAttention` with optional key/value cache updates during decoding.

```python
import tensorflow as tf
import mlable.layers.transformer
import mlable.caching

batch_dim, seq_dim, embed_dim, num_heads, head_dim = 1, 4, 6, 2, 3
layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=num_heads, key_dim=head_dim)

x = tf.ones((batch_dim, 1, embed_dim), dtype=tf.float32)
cache = mlable.caching.create(batch_dim=batch_dim, cache_dim=seq_dim, num_heads=num_heads, head_dim=head_dim)
out, scores, next_cache = layer(
    query=x,
    value=x,
    cache=cache,
    step=2,
    return_attention_scores=True,
)
```

Call-time additions relative to base MHA:

- `cache`: optional stacked key/value cache.
- `step`: optional update index (`None` appends).

See `tests/layers/test_transformer.py` for batch and sequential decode usage.
