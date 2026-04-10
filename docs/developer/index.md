# Developer Guide

This guide addresses contributors and library developers. It covers the project's coding and testing conventions, the established extension points at the tool layer, and the deeper architectural patterns governing the core data model and force field hierarchy.

## Part I: Project Conventions

The following pages document day-to-day development practices.

- [Development Setup](development-setup.md) — repository cloning, editable installation, and test execution
- [Contributing](contributing.md) — the pull request workflow, commit message conventions, and the pre-commit hook suite
- [Coding Style](coding-style.md) — naming conventions, formatting requirements, and the immutability constraint
- [Testing](testing.md) — pytest conventions, test markers, coverage requirements, and the distinction between local and external tests
- [Release Process](release-process.md) — semantic versioning, changelog maintenance, and CI-driven package publication

## Part II: Extending the Tool Layer

The extension points documented here present stable interfaces with explicit contracts. Contributing a new capability at this layer requires implementing a subclass or registering a handler; no modification of core internals is necessary.

- [Adding a Tool or Compute Operation](extending-tools.md) — the `Tool` and `Compute` protocols, operator registration, and custom preparation recipes
- [Adding an I/O Format](extending-io.md) — reader and writer base classes, the `FieldFormatter` canonicalization interface, and force field formatter registration
- [Adding a Wrapper or Adapter](extending-integration.md) — subprocess wrapper conventions for external CLI tools and in-memory adapter patterns for third-party libraries

## Part III: Extending Core Structures

Modifications at this layer require a thorough understanding of MolPy's type bucket system, force field parameter hierarchy, and potential dispatch chain. Contributions should be discussed in a GitHub issue before implementation begins.

- [Extending the Data Model](extending-core.md) — new `Entity` and `Link` subtypes, custom `Struct` subclasses, and identity-based hashing invariants
- [Extending the Force Field](extending-forcefield.md) — custom `Style`, `Type`, and `Potential` definitions, and `ForceFieldFormatter` registration for serialization

## Integrations

- [MCP Server for LLM Agents](../user-guide/mcp.md) — exposing MolPy's source code and symbol index to large language model agents via the Model Context Protocol

## Issue Tracking and Discussion

- Bug reports and feature requests: <https://github.com/MolCrafts/molpy/issues>
- Design discussions: <https://github.com/MolCrafts/molpy/discussions>
