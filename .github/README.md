# MLable

TensorFlow utilities for layers, metrics, shaping, sampling, and model building blocks.

## Installation

```bash
pip install -U mlable
```

## Quickstart

```python
import tensorflow as tf
import mlable.layers.shaping
import mlable.metrics

x = tf.ones(shape=(2, 4, 8))
y = mlable.layers.shaping.Divide(axis=-1, factor=4, insert=True)(x)

metric = mlable.metrics.CategoricalGroupAccuracy(group=4)
score = metric(y_true=tf.one_hot([[1, 2, 3, 4]], depth=8), y_pred=tf.one_hot([[1, 2, 0, 4]], depth=8))
```

## Documentation

Detailed usage documentation is available under `docs/sources/`:

- [`docs/sources/layers/shaping.md`](../docs/sources/layers/shaping.md)
- [`docs/sources/layers/embedding.md`](../docs/sources/layers/embedding.md)
- [`docs/sources/layers/transformer.md`](../docs/sources/layers/transformer.md)
- [`docs/sources/metrics/group_accuracy.md`](../docs/sources/metrics/group_accuracy.md)
- [`docs/sources/index.md`](../docs/sources/index.md)

## Credits

[Andrej Karpathy][video-karpathy] reconnected my ML synapses with [micrograd][code-micrograd].

## License

Licensed under the [aGPLv3](LICENSE.md).

[code-micrograd]: https://github.com/karpathy/micrograd
[video-karpathy]: https://www.youtube.com/@AndrejKarpathy/videos
