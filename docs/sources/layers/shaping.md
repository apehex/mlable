# `mlable.layers.shaping`

Keras layers that reshape tensors without changing values.

## `Divide`

Split one axis by a factor.

```python
import tensorflow as tf
import mlable.layers.shaping

x = tf.ones(shape=(2, 4, 8))
layer = mlable.layers.shaping.Divide(axis=-1, factor=4, insert=True, right=True)
y = layer(x)

list(y.shape)
# [2, 4, 2, 4]
```

Arguments:

- `axis`: axis to divide.
- `factor`: division factor.
- `insert`: if `True`, inserts a new axis; otherwise updates an existing one.
- `right`: controls where the new/split axis is placed.

## `Merge`

Merge two consecutive axes into one.

```python
import tensorflow as tf
import mlable.layers.shaping

x = tf.ones(shape=(2, 4, 2, 4))
layer = mlable.layers.shaping.Merge(axis=-1, right=False)
y = layer(x)

list(y.shape)
# [2, 4, 8]
```

Arguments:

- `axis`: reference axis used for merge.
- `right`: controls which side is merged into the other.

## Notes from tests

- `Divide` and `Merge` are reciprocal in common cases.
- Reshaping keeps flattened values unchanged.
- Output shapes follow the constraints validated in `tests/layers/test_shaping.py`.
