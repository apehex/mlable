---
description: Interview-first task intake and planning.
---

# Agent Guidelines

Guidelines for LLM agents working on this repository.

## Interview-First Workflow

- For each new task begin by clarifying intent before implementation.
- After clarification, briefly restate the understood request and propose a plan.
- Do not ask redundant questions when the context is already clear.
- Do not jump straight to implementation for ambiguous or strategic work.

## Working Rules

- Keep track of the long planning and goals in `docs/roadmap.md`.
- Split complex tasks into incremental and focused commits.
- Read relevant source and tests before editing code.
- Avoid broad refactors unless required by the issue.
- If tooling is unavailable in the environment, record that limitation in your notes.

## Coding Rules

- Preserve existing package and test structure unless asked otherwise.
- Prefer factual documentation updates over speculative wording.
- Think through edge cases, tradeoffs, dependencies, and failure modes before coding.
- Run the existing test command(s) available in the environment before and after changes when possible.

## Documentation Rules

- Keep `docs/index.md` aligned with the current documentation tree.
- Keep `docs/sources/index.md` aligned with current modules in `src/mlable/`.
- Store proposals in `ideas.md`, selected work in `todo.md`, and curated GitHub issue focus in `issues.md`.
