# Coding Style

MolPy follows a “clean, explicit, type‑driven” coding style designed to make
the library easy to read, test and extend.

This page summarizes the main expectations for contributors.

---

## General Python style

- Follow **PEP 8** where it makes sense.
- Prefer **readability** over cleverness.
- Use **snake_case** for functions and variables, **PascalCase** for classes.
- Avoid mutable global state and singletons.

---

## Types and imports

- Use **type hints everywhere**:
  - Prefer `list[T]` / `dict[K, V]` over `List` / `Dict`.
  - Avoid `Any` unless absolutely necessary.
  - Use `Literal[...]` when helpful for clarity.
- Keep imports **explicit and at the top** of the file.
- Avoid wildcard imports (`from x import *`).

---

## Core data structures

The core types (`Frame`, `Block`, `Box`, `Atomistic`, wrappers) should:

- Behave like **value‑like containers** with minimal side effects.
- Avoid engine‑specific or format‑specific logic.
- Be small and composable – heavy logic belongs in helper modules.

If you find yourself adding complex behavior directly into these core classes,
consider moving it into:

- A separate utility function, or
- A higher‑level module that consumes the core types.

---

## Error handling and logging

- Validate inputs early and fail fast with clear `ValueError` / `TypeError`
  messages.
- Use the library’s logging utilities where appropriate rather than printing
  directly.
- Do not silently swallow exceptions unless there is a very good reason,
  and document that behavior.

---

## Tests and examples

- Every non‑trivial feature should be covered by tests in `tests/`.
- Prefer small, realistic molecular systems in tests (water, methane, short
  chains) rather than arbitrary random data.
- When adding new features, consider adding a short example to the docs.

Keeping tests and docs close to the real API is the best way to avoid drift.
