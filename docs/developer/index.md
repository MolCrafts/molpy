# Developer Guide

This guide addresses contributors and library developers working on MolPy as a research software library. It is organized around what you came here to do:

- **Fix a bug or land a first PR** → [Development Setup](development-setup.md), then [Contributing Workflow](contributing.md) and [Testing](testing.md)
- **Add an analysis operation, file format, tool integration, or force-field style** → the plug-in recipes under [Extending MolPy](extending-compute.md) and [Extending the Force Field](extending-forcefield.md); no core changes required
- **Change typification, the data model, or force-field internals** → read the [Architecture Overview](architecture-overview.md) first, open a GitHub issue to discuss, then follow [Extending Typifiers](extending-typifiers.md) or [Extending the Data Model](extending-core.md)
- **Understand how MolPy is put together** → [Architecture Overview](architecture-overview.md) and [molrs Backend](molrs-backend.md)
- **Cut a release** → [Release Process](release-process.md)

Canonical field names and topology keys live in the [Naming Conventions](../tutorials/naming-conventions.md) appendix of the Tutorials.

## Contributing

Day-to-day development practices:

- [Development Setup](development-setup.md) — repository cloning, editable installation, building molrs from source, and test execution
- [Contributing Workflow](contributing.md) — the pull request workflow, commit message conventions, and the pre-commit hook suite
- [Coding Style](coding-style.md) — identifier style, formatting requirements, and the mutation contract
- [Testing](testing.md) — pytest conventions, test markers, coverage requirements, and the distinction between local and external tests
- [Release Process](release-process.md) — the shared molpy/molrs minor line and CI-driven package publication

## Architecture

The design context that the extension recipes assume:

- [Architecture Overview](architecture-overview.md) — module responsibilities, the graph and tabular layers, field-name translation, the mutation contract, and the performance model of assembly
- [molrs Backend](molrs-backend.md) — how the Rust column store and compute kernels surface in Python: boxes, neighbor lists, `Rdf`, and the analysis catalog

## Extending MolPy

Ordered from plug-in interfaces (implement a subclass, register a handler) to core internals (discuss in a GitHub issue before implementation):

- [Adding a Compute Operation](extending-compute.md) — the `Compute` protocol for reusable analysis operations
- [Adding an I/O Format](extending-io.md) — native `read_X` / `write_X` doors re-exported on `mp.io`, and when molpy adds a function of its own
- [Adding a Wrapper or Adapter](extending-integration.md) — subprocess wrapper conventions and in-memory adapter patterns
- [Extending Typifiers](extending-typifiers.md) — the `assign` → `TypeAssignment` contract, matcher boundaries, and where a typifier lives (native vs. tool-driven)
- [Extending the Data Model](extending-core.md) — why a new node or relation kind is a molrs change, and what Python may add on top
- [Extending the Force Field](extending-forcefield.md) — a new style or category registered from Python, no rebuild

## Issue Tracking and Discussion

- Bug reports and feature requests: <https://github.com/MolCrafts/molpy/issues>
- Design discussions: <https://github.com/MolCrafts/molpy/discussions>
