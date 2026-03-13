# Context

## Project

- Name: `mlable`
- Package focus: TensorFlow utilities for layers, model blocks, metrics, sampling, schedules, and shaping helpers.
- Packaging: Poetry project (`pyproject.toml`) targeting Python `>=3.10,<4.0`.

## Source Layout

- `mlable/blocks`: Attention and convolution building blocks, plus normalization and patching layers.
- `mlable/layers`: Embedding, transformer, and shape-manipulation layers.
- `mlable/maths`: Tensor reduction and probability helpers.
- `mlable/models`: Generic contrast model, VAE model, and diffusion model classes.
- `mlable/shaping`: Axis operations and Hilbert-curve folding utilities.
- Top-level modules: caching, data, masking, metrics, sampling, schedules, shapes, text, utils.

## Test Layout

- Tests are in `tests/`.
- The test tree mirrors source packages (`blocks`, `layers`, `maths`, `models`, `shaping`) plus top-level module tests.
