# Adapter & Wrapper

This section documents MolPy’s **external integration layers**.

Use these layers to integrate optional tooling (CLIs or third-party libraries) without pushing those dependencies into the core data model.

## Concepts

- `molpy.adapter`: representation adapters (bidirectional sync; **no subprocess execution**)
- `molpy.wrapper`: subprocess wrappers for external CLIs/binaries (explicit side effects)

If you are looking for the *design rules and patterns* (what belongs where), see:

- Developer guide: `Developer → Wrapper & Adapter`
- Tutorial: `Tutorials → External Tools`

## Adapter API

::: molpy.adapter

## Wrapper API

::: molpy.wrapper

## Compatibility facade

`molpy.external` remains as a temporary compatibility facade that re-exports the new adapter/wrapper APIs.

New code should import directly from `molpy.adapter` / `molpy.wrapper`.

::: molpy.external
