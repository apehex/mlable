# MLable

Tensorflow libs:

- [layers](#layers):
    - reshaping:
        - [Divide](#divide)
        - [Merge](#merge)
    - embedding:
        - [RotaryPositionalEmbedding](#RotaryPositionalEmbedding)
    - transformer:
        - [CachedMultiHeadAttention](#CachedMultiHeadAttention)
        - [FeedForwardGate](#FeedForwardGate)
- [metrics](#layers):
    - [CategoricalGroupAccuracy](#CategoricalGroupAccuracy)

## Installation

The package is available on pypi:

```python
pip install -U mlable
```

## Layers

### Divide

Relative reshaping layers that divides a given axis and multiplies another by the same factor:

```python
import mlable.layers.reshaping

__x = tf.ones(shape=(2, 4, 6, 8))
__l = mlable.layers.reshaping.Divide(
    input_axis=2, # relative to the NEW shape / rank
    output_axis=-1, # same
    factor=3,
    insert=False,) # whether to create a new axis

list(__l(__x).shape)
# [2, 4, 2, 24]
```

### Merge

Relative reshaping layers that merges two axes:

```python
import mlable.layers.reshaping

__x = tf.ones(shape=(2, 4, 6, 8))
__l = mlable.layers.reshaping.Merge(
    left_axis=1,
    right_axis=-1,
    left=False,) # whether to merge into the left axis

list(__l(__x).shape)
# [2, 6, 32]
```

### CachedMultiHeadAttention

This layer subclasses the regular [MultiHeadAttention][docs-tf-multiheadattention] and adds a cache.

It has the same parameters:

```python
import mlable.layers.transformer

mlable.layers.transformer.CachedMultiHeadAttention(
    num_heads,
    key_dim,
    value_dim=None,
    dropout=0.0,
    use_bias=True,
    output_shape=None,
    attention_axes=None,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs)
```

And its `call` function has the following arguments:

```python
mlable.layers.transformer.CachedMultiHeadAttention.call(
    query,
    value,
    key=None,
    cache=None,
    step=None,
    training=False,
    attention_mask=None,
    return_attention_scores=False,
    use_causal_mask=True,)
```

### FeedForwardGate

A typical feed-forward layer with GELU activation:

```python
import mlable.layers.transformer

__x = tf.ones(shape=(2, 3, 5), dtype=tf.dtypes.float32)
__l = mlable.layers.transformer.FeedForwardGate(
    input_dim=256,
    hidden_dim=1024)

__l(__x)
```

### RotaryPositionalEmbedding

Tensorflow implementation of [RoPE][arxiv-rope]:

```python
import mlable.layers.embedding

__x = tf.ones(shape=(2, 3, 5))
__l = mlable.layers.embedding.RotaryPositionalEmbedding(
    sequence_axis=1, # position along this axis
    feature_axis=-1, # output axis
    max_wavelength=10_000, # see the paper
    scaling_factor=1.) # see the paper

__l(inputs=__x, offset=2) # the offset is typically used to perform iterative decoding during inference
```

## Metrics

### CategoricalGroupAccuracy

Hierarchical models should not be scored on individual predictions but on their combination.

For example, [tokun][github-tokun] is a byte level autoencoder.
It predicts probabilities for each byte of the output, like 0 in the UTF-32-BE encoding of "a" `(0, 0, 0, 97)`.

A prediction of `(0, 0, 0, 98)` for "a" has 3 correct byte out of 4, but the prediction is actually "b".

In this case the byte accuracy is 75% while the character accuracy is 0%.
Having several hierarchies of scoring helps with training and evaluation.

```python
import mlable.metrics

byte_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=1, name='byte_accuracy')
character_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=4, name='character_accuracy')
token_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=64, name='token_accuracy')
```

## Credits

[Andrej Karpathy][video-karpathy] reconnected my ML synapses with [micrograd][code-micrograd].

## License

Licensed under the [aGPLv3](LICENSE.md).

[arxiv-rope]: https://arxiv.org/pdf/2104.09864
[code-micrograd]: https://github.com/karpathy/micrograd
[docs-tf-multiheadattention]: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
[github-tokun]: https://github.com/apehex/tokun
[video-karpathy]: https://www.youtube.com/@AndrejKarpathy/videos
