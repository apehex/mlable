# Index

Overview of the repository structure.

## Docs

| Path                      | Purpose                               |
| ------------------------- | ------------------------------------- |
| `-- docs/`                | Collaboration-focused project docs    |
| `   -- agents.md`         | Guidelines for LLM agents             |
| `   -- context.md`        | Overview of the project               |
| `   -- decisions.md`      | Record of important design choices    |
| `   -- ideas.md`          | Research directions and speculations  |
| `   -- index.md`          | Structure of the repository           |
| `   -- invariants.md`     | Hard constraints                      |
| `   -- issues.md`         | Curated list of selected issues       |
| `   -- package/`          | Package/module documentation          |
| `      -- index.md`       | Index of package modules              |
| `      -- layers/`        | Layer usage guides                    |
| `      -- metrics/`       | Metric usage guides                   |
| `   -- primer.jj`         | Conversation primer (Jinja template)  |
| `   -- references.md`     | External references                   |
| `   -- todo.md`           | Concrete next tasks                   |

## Sources

| Path                      | Purpose                               |
| ------------------------- | ------------------------------------- |
| `-- src/mlable`           | Root of the Python package            |
| `   -- blocks`            | Reusable model blocks                 |
| `   -- layers`            | Standalone Keras layers               |
| `   -- maths`             | Math and probability helpers          |
| `   -- models`            | Generic model wrappers                |
| `   -- shaping`           | Axis and spatial transforms           |

## Tests

| Path                      | Purpose                               |
| ------------------------- | ------------------------------------- |
| `-- tests`                | Unit tests mirroring package layout   |
