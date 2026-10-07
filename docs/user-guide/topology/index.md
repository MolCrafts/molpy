# Polymer Topologies

**From one ethylene-oxide unit to chains, rings, stars and combs: a topology is a CGsmiles string.**

Every architecture here is a **short guide page** paired with a **runnable script** of the same name under `examples/topology/`. They all compose the same three native primitives:

1. **Units** — a CGsmiles fragment per repeat unit, with bonding descriptors as its **ports** (`[<]OCC[>]`), turned into a 3D `mp.Atomistic` carrying its ports by `mp.conformer.Conformer`.
2. **Topology** — a CGsmiles string (`{[#EO]|10}`), turned into a **site graph** by `mp.io.cgsmiles.CgSmilesIr(...).to_coarsegrain()`: one site per unit, one bond per join.
3. **Assembly** — `mp.builder.Assembler(library, mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)`: one copy of `library[bead_type]` per site, each bond joining one port of each end, handed back as the graph class you name.

```bash
cd examples
python topology/01_linear.py
```

## How the pages and scripts line up

| Guide | Example script | Architecture |
|-------|----------------|--------------|
| [Linear](01_linear.md) | `examples/topology/01_linear.py` | Homopolymer path |
| [Block / sequence](02_block.md) | `examples/topology/02_block.py` | Copolymer sequence |
| [Macrocycle](03_ring.md) | `examples/topology/03_ring.py` | Closed ring of units |
| [Star](04_star.md) | `examples/topology/04_star.py` | Multifunctional core + arms |
| [Comb](05_comb.md) | `examples/topology/05_comb.py` | Backbone with labelled graft ports |
| [Telechelic](06_telechelic.md) | `examples/topology/06_telechelic.py` | Capped chain ends |

Shared chemistry: `examples/topology/eo_kit.py` · examples index:
`examples/topology/README.md`.

## Ports decide who joins whom

A bond between two sites joins one port of each unit, and the two ports must **accept** each other:

| Port | Joins |
|------|-------|
| `<` | a `>` with the same label |
| `>` | a `<` with the same label |
| `$` | a `$` with the same label |

A label (`[>g]`, `[<g]`) restricts a port to partners with the same label; the [comb](05_comb.md) uses one to keep grafts off the backbone. The port's leaving group (the hydrogen a conformer adds on it) is removed when the bond forms, and ports left without a bond keep theirs, as a chain's end ports do.

The assembler chooses the ports itself: it walks each molecule from one end, gives every bond of a site one distinct port that its partner can accept, and retries from the other end of a chain that it entered backwards.

## Growing without coordinates

A CGsmiles topology has no coordinates, so the examples use `mp.builder.GrowthPlacer`: the first unit keeps its conformer pose and every later unit is joined onto its parent's port, pointing back along the parent's leaving bond. Bond lengths, ring closures and overlaps are left to a later minimisation. With coordinates, for example a site graph coarsened from a CG model by `mp.builder.Coarsener`, use `mp.builder.SitePlacer` and `mp.builder.AxisOrienter` instead.

## The kit (`eo_kit.py`)

| Unit | CGsmiles | Role |
|------|----------|------|
| `EO` | `[<]OCC[>]` | –O–CH₂–CH₂– repeat unit |
| `PO` | `[<]OC(C)C[>]` | –O–CH(CH₃)–CH₂– repeat unit |
| `CAPA` / `CAPB` | `C[>]` / `[<]OC` | methyl / methoxy chain ends |
| `X3` | `C(C[>])(C[>])C[>]` | three-arm core |
| `BR` / `GR` | `[<]OCC(C[>g])[>]` / `[<g]OCC[>]` | comb branch point / first graft unit |

## Networks

Crosslinked gels and networks are not covered: joining sites by proximity is not yet a site-graph primitive.

## See also

- [Packing Systems](../09_packing.md) — full cells from single chains
- [Builder API](../../api/builder.md)
