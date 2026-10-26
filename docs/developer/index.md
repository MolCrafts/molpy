# Developer Guide

This guide addresses contributors and library developers working on MolPy as a research software library. It is organized around what you came here to do:

- **Fix a bug or land a first PR** → [Development Setup](development-setup.md), then [Contributing Workflow](contributing.md) and [Testing](testing.md)
- **Add an analysis operation, file format, tool integration, or force-field style** → the recipes under [Extending MolPy](extending-compute.md) and [Extending the Force Field](extending-forcefield.md); a one-off analysis or a new force-field style needs no Rust change, a shipped analysis or file format goes into the Rust core
- **Change typification, the data model, or force-field internals** → read the [Architecture Overview](architecture-overview.md) first, open a GitHub issue to discuss, then follow [Extending Typifiers](extending-typifiers.md) or [Extending the Data Model](extending-core.md)
- **Understand how MolPy is put together** → [Architecture Overview](architecture-overview.md) and [The Rust Core](rust-core.md)
- **Cut a release** → [Release Process](release-process.md)

Canonical field names and topology keys live in the [Naming Conventions](../tutorials/naming-conventions.md) appendix of the Tutorials.

## Contributing

Day-to-day development practices:

- [Development Setup](development-setup.md) — repository layout, the editable install that builds the Rust core from source, the gates, and test execution
- [Contributing Workflow](contributing.md) — the pull request workflow, commit message conventions, and the pre-commit hook suite
- [Coding Style](coding-style.md) — identifier style, formatting requirements, and the mutation contract
- [Testing](testing.md) — where the Rust unit tests and the pytest suite live, what is and is not a test, and why no test needs an external program
- [Release Process](release-process.md) — one version for the whole project and CI-driven publication (releases are currently frozen)

## Architecture

The design context that the extension recipes assume:

- [Architecture Overview](architecture-overview.md) — module responsibilities, the graph and tabular layers, field-name translation, the mutation contract, and the performance model of assembly
- [The Rust Core](rust-core.md) — how the Rust column store and compute kernels surface in Python: boxes, neighbor lists, `Rdf`, and the analysis catalog

## Extending MolPy

Ordered from plug-in interfaces (implement a subclass, register a handler) to core internals (discuss in a GitHub issue before implementation):

- [Adding a Compute Operation](extending-compute.md) — the `Compute` protocol for reusable analysis operations
- [Adding an I/O Format](extending-io.md) — native `read_X` / `write_X` doors re-exported on `mp.io`, and why Python adds no reader of its own
- [Extending Typifiers](extending-typifiers.md) — the `assign` → `TypeAssignment` contract, matcher boundaries, and where a typifier lives (native vs. tool-driven)
- [Extending the Data Model](extending-core.md) — why a new node or relation kind is a Rust-core change, and what Python may add on top
- [Extending the Force Field](extending-forcefield.md) — a new style or category registered from Python, no rebuild

## Issue Tracking and Discussion

- Bug reports and feature requests: <https://github.com/MolCrafts/molpy/issues>
- Design discussions: <https://github.com/MolCrafts/molpy/discussions>
