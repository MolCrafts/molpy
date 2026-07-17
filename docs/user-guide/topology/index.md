# Polymer Topologies

**From one ethylene-glycol monomer to linear chains, stars, gels, and dual networks — topology is only a pairing rule.**

This section is a self-contained lab manual for MolPy’s polymer assembly stack. [Assembly](../02_assembly.md) explains the kernel (`GraphAssembler`, selectors, charge freezing). Here every architecture is a **short guide page** paired with a **runnable script** of the same name under `examples/topology/`.

```bash
cd examples
python topology/01_linear.py
python topology/run_all.py    # smoke the full catalogue
```

## How the pages and scripts line up

| Guide | Example script | Architecture |
|-------|----------------|--------------|
| [Linear](01_linear.md) | `examples/topology/01_linear.py` | Homopolymer path |
| [Block / sequence](02_block.md) | `examples/topology/02_block.py` | Copolymer sequence |
| [Macrocycle](03_ring.md) | `examples/topology/03_ring.py` | Residue ring |
| [Star](04_star.md) | `examples/topology/04_star.py` | Multifunctional core + arms |
| [Comb](05_comb.md) | `examples/topology/05_comb.py` | Backbone grafts |
| [Telechelic](06_telechelic.md) | `examples/topology/06_telechelic.py` | CAPA–EO–CAPB |
| [Exhaustive gel](07_gel_exhaustive.md) | `examples/topology/07_gel_exhaustive.py` | Distance mesh |
| [Random gel](08_gel_random.md) | `examples/topology/08_gel_random.py` | Conversion + seed |
| [End-linked](09_end_linked.md) | `examples/topology/09_end_linked.py` | Ends only |
| [Dual network](10_dual_network.md) | `examples/topology/10_dual_network.py` | Two assemble steps |
| [Prepolymer + agent](11_prepolymer_agent.md) | `examples/topology/11_prepolymer_agent.py` | Agent cure |

Shared chemistry: `examples/topology/eo_kit.py` · examples index:
`examples/topology/README.md`.

Layout (docs ↔ examples, same basenames):

```text
docs/user-guide/topology/          examples/topology/
  index.md                           README.md · eo_kit.py · run_all.py
  01_linear.md                       01_linear.py
  02_block.md                        02_block.py
  …                                  …
  11_prepolymer_agent.md             11_prepolymer_agent.py
```

## The three decisions

Whatever architecture you want, MolPy asks for the same three answers.

1. **Which atoms may react** — `fields.SITE` via `SiteMap`
2. **What the reaction does** — `mp.Reaction(...)` (Daylight reaction SMARTS)
3. **Which sites pair** — a `Selector` (chosen for you by `build_*`, or passed to `assemble`)

**`PolymerBuilder.build(cgsmiles)` is the only expand + assemble entry.**
`build_linear`, `build_sequence`, `build_ring`, and `build_star` only format CGSmiles and call `build`.

```python
builder.build_linear("EO", 5)   # identical to:
builder.build("{[#EO]|5}")
```

Statistical networks skip `build`’s residue graph: you already have a world, then

```python
GraphAssembler(reaction).assemble(world, ExhaustiveSelector(...))
# or RandomSelector / SpacingSelector / ExplicitPairSelector
```

## Reaction SMARTS (one screen)

Reactants left of `>>`, products right. Atom maps (`:1`) preserve identity; unmapped left-hand atoms are **leaving groups**.

**Ether condensation** (ruled topologies + agent cure):

```text
[O;%a:1][H] . [C:2][O;%b][H]  >>  [O:1][C:2]
```

| Piece | Meaning |
|-------|---------|
| `%a` / `%b` | only match atoms with `fields.SITE == "a"` / `"b"` |
| `:1`, `:2` | same atom on both sides of `>>` |
| unmapped H | deleted |

**C–C crosslink** (after marking carbons `x` and leaving H `h`):

```text
[C;%x:1][H;%h] . [C;%x:2][H;%h]  >>  [C:1][C:2]
```

```python
from molpy.builder.assembly import SiteMap
SiteMap(eo).label_elements("O", "a", "b")
```

## The kit (`eo_kit.py`)

| Template | Role | Sites |
|----------|------|-------|
| `EO` | Bifunctional glycol (OCCO) | `a`, `b` on hydroxyl O |
| `CAPA` / `CAPB` | Monofunctional caps | only `a` or only `b` |
| `X3` | Star core (glycerol-like) | three `a` |
| `X4` | Tetrafunctional agent | four `a` |
| `BR` | Comb junction | `a`, `a`, `b` |

| Constant | Use |
|----------|-----|
| `ETHER` | Main-chain / agent condensation |
| `XLINK` | First network (`x` / `h`) |
| `XLINK2` | Second network (`y` / `k`) |

## Ruled vs statistical

| Kind | Entry | Pairing |
|------|-------|---------|
| Linear, block, ring, star, comb, telechelic | `PolymerBuilder.build` / `build_*` | `TopologySelector` (residue edges) |
| Gel, end-link, dual net, agent cure | `GraphAssembler.assemble` | Proximity selectors |

Complex materials are **stacks** of these edits (build → mark → `Replicas` → assemble → …), not new builder classes.

## Production GAFF gel

Teaching scripts stay pure assembly. A parameterized offline gel (AmberTools + LAMMPS) is documented in [Building a Crosslinked Gel](../16_crosslinked_gel.md) and implemented under `polymer_builders/peo_gel/`.

## See also

- [Assembly](../02_assembly.md) — kernel details
- [Polydisperse Systems](../05_polydisperse_systems.md) — sample lengths → `build_sequence`
- [Packing Systems](../09_packing.md) — full cells beyond `Replicas`
- [Builder API](../../api/builder.md)
