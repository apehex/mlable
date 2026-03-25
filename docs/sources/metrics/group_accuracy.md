# `mlable.metrics` grouped accuracies

Grouped accuracies score logical entities made of multiple elementary predictions.

## `CategoricalGroupAccuracy`

For one-hot or categorical logits.

```python
import mlable.metrics

byte_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=1, name='byte_accuracy')
character_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=4, name='character_accuracy')
```

Arguments:

- `group`: group size (or list of group sizes).
- `axis`: reduction axis (or list of axes).
- `depth`: category depth when predictions are flattened.

## `BinaryGroupAccuracy`

For binary-encoded predictions.

```python
import mlable.metrics

byte_accuracy = mlable.metrics.BinaryGroupAccuracy(group=1, depth=8, name='byte_accuracy')
character_accuracy = mlable.metrics.BinaryGroupAccuracy(group=4, depth=8, name='character_accuracy')
```

Arguments:

- `group`: group size (or list of group sizes).
- `axis`: reduction axis (or list of axes).
- `depth`: binary depth when predictions are flattened.
- `from_logits`: whether `y_pred` is treated as logits.

## `RawGroupAccuracy`

For raw numeric values after scaling.

```python
import mlable.metrics

byte_accuracy = mlable.metrics.RawGroupAccuracy(group=1, factor=256.0, name='byte_accuracy')
character_accuracy = mlable.metrics.RawGroupAccuracy(group=4, factor=256.0, name='character_accuracy')
```

Arguments:

- `group`: group size (or list of group sizes).
- `axis`: reduction axis (or list of axes).
- `factor`: scaling factor applied before comparison.

## Notes from tests

- Accuracy stays within `[0, 1]`.
- Larger groups are stricter than byte-level groups.
- Multiple group reductions are supported for categorical metrics.

See `tests/test_metrics.py` for concrete expected values.
