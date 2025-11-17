# Contributing

Thanks for your interest in contributing to MolPy!

We aim to keep the codebase:

- **Predictable** – clear data model, minimal surprises
- **Well‑tested** – every feature backed by tests
- **Documented** – user‑facing features have docs and examples

---

## How to propose changes

1. **Open an issue** describing:
   - What you want to change
   - Why it’s useful
   - Any alternatives you considered
2. **Fork the repo** and create a feature branch.
3. Implement the change with:
   - Type hints
   - Tests (pytest)
   - Documentation updates where relevant
4. Open a **pull request** against the main branch.

Small, focused PRs are much easier to review than large, multi‑topic ones.

---

## Code style and quality

- Follow standard Python style (PEP 8) with **type hints everywhere**.
- Prefer explicit, functional code over clever one‑liners.
- Keep core data structures (`Frame`, `Atomistic`, `Box`, …) free from
  engine‑specific or UI‑specific logic.

See `developer/coding-style.md` for more details.

---

## Tests

- Use **pytest** for all tests under `tests/`.
- New features should come with new tests; bug fixes should include regression tests.
- Try to use small, realistic molecular systems (water, methane, small polymers)
  for test cases instead of synthetic random data where possible.

See `developer/testing.md` for more guidance.

---

## Documentation

- User‑facing features should be mentioned in:
  - `getting-started` (if broadly relevant), or
  - `user-guide/*` / `concepts/*` / `tutorials/*` as appropriate.
- Keep examples short and runnable.
- Make sure docs and code **stay in sync** (no pseudo‑APIs).

If you’re unsure where to document something, mention it in your PR and we’ll
help place it.

---

## Release notes

When you add user‑visible features or behavior changes, please:

- Add a brief entry to the unreleased section of `changelog/releases.md`
  (or create one if needed).

This keeps the changelog useful for users and downstream projects.
