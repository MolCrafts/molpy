# Demo models

`aspirin.glb` is produced by **molvis's own glTF exporter** — the same
ball-and-stick geometry the engine renders, serialized to a self-contained
glTF that needs zero molvis runtime to view. It is **not** hand-written
geometry.

- Source structure: `aspirin.sdf` (PubChem CID 2244, 3D conformer).
- Exporter: `exportFrameToGLB` / `MolvisApp.exportGLTF` in `@molcrafts/molvis-core`.

Regenerate (molvis is a sibling checkout at `../molvis`):

```sh
cd ../molvis
npx tsx core/scripts/gen-molecule-glb.mts \
  ../molpy/docs/assets/models/aspirin.sdf \
  ../molpy/docs/assets/models/aspirin.glb
```

Colours (CPK), radii, split-coloured bonds, and bond orders come straight
from molvis's render buffers, so the exported model matches what molvis draws.
The material is matte PBR (metallic 0, roughness 0.9).
