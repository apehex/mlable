# MLable

Tensorflow libs:

- [layers](#layers):
    - reshaping:
        - [Divide](#divide)
        - [Merge](#merge)
    - embedding:
        - [TokunEmbedding](#TokunEmbedding)
        - [RotaryPositionalEmbedding](#RotaryPositionalEmbedding)
    - transformer:
        - [CachedMultiHeadAttention](#CachedMultiHeadAttention)
        - [FeedForwardGate](#FeedForwardGate)
- [metrics](#layers):
    - [BinaryGroupAccuracy](#BinaryGroupAccuracy)
    - [CategoricalGroupAccuracy](#CategoricalGroupAccuracy)
    - [RawGroupAccuracy](#RawGroupAccuracy)

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

### TokunEmbedding

These embeddings are made from the combination of elementary embeddings.

The layer inherits from `keras.layers.Embedding`.
It expects a tensor with a shape following the structure:

- axis `-2`: sequence axis, with dimension `S / T`
- axis `-1`: token axis, with dimension `T`

The `T` values in the token axis are the indexes of the embeddings to be combined.
Typically, these are byte values:

```python
import mlable.layers.embedding

__x = tf.random.uniform((128, 1024, 16), minval=0, maxval=256, dtype=int32)
__l = mlable.layers.embedding.TokunEmbedding(
    input_dim=256,
    output_dim=128,)

list(__l(__x).shape)
# [128, 1024, 2048]
```

And the output tensor has a shape `(..., S / T, T * E)`, where `T * E = H` is the embedding dimension inside the LLM.
In the above example, it is set to 2048.

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

## Metrics

Hierarchical models should not be scored on individual predictions but on their combination.

For example, [tokun][github-tokun] is a byte level autoencoder.
It predicts probabilities for each byte of the output, like 0 in the UTF-32-BE encoding of "a" `(0, 0, 0, 97)`.

A prediction of `(0, 0, 0, 98)` for "a" has 3 correct byte out of 4, but the prediction is actually "b".

In this case the byte accuracy is 75% while the character accuracy is 0%.
Having several hierarchies of scoring helps with training and evaluation.

The individual predictions are evaluated in groups forming logical entities.
These predictions can be in binary, categorical or raw formats.
Each of these formats has a dedicated metric.

### BinaryGroupAccuracy

Arguments:

- `group`: the number of elementary predictions that must be correct to predict a higher level entity
- `depth`: the dimension of the binary embedding for each predicted value
- `threshold`: probabilities below the threshold are scored as `0` and above `1`

```python
import mlable.metrics

byte_accuracy = mlable.metrics.BinaryGroupAccuracy(group=1, depth=8, threshold=0.6, name='byte_accuracy')
character_accuracy = mlable.metrics.BinaryGroupAccuracy(group=4, depth=8, threshold=0.6, name='character_accuracy')
token_accuracy = mlable.metrics.BinaryGroupAccuracy(group=64, depth=8, threshold=0.6, name='token_accuracy')
```

### CategoricalGroupAccuracy

Arguments:

- `group`: the number of elementary predictions that must be correct to predict a higher level entity

```python
import mlable.metrics

byte_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=1, name='byte_accuracy')
character_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=4, name='character_accuracy')
token_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=64, name='token_accuracy')
```

### RawGroupAccuracy

Arguments:

- `group`: the number of elementary predictions that must be correct to predict a higher level entity
- `factor`: scaling factor, typically from a probability distribution to a numeric value

```python
import mlable.metrics

byte_accuracy = mlable.metrics.RawGroupAccuracy(group=1, factor=256.0, name='byte_accuracy')
character_accuracy = mlable.metrics.RawGroupAccuracy(group=4, factor=256.0, name='character_accuracy')
token_accuracy = mlable.metrics.RawGroupAccuracy(group=64, factor=256.0, name='token_accuracy')
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
