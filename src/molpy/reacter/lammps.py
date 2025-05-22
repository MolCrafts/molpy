from pathlib import Path
import molpy as mp
from collections import deque

def _collect_atoms_index_between_to_site(graph, init, edge):
    """
    Collect all atom indices between two sites (init and edge) in the graph.

    Args:
        graph: Topology object with 'atoms' and 'bonds'.
        init: Index of the starting atom.
        edge: Index of the ending atom.

    Returns:
        List of atom indices (in graph.atoms) that are on the path from init to edge, including both.
    """
    # Build adjacency list
    n_atoms = len(graph.atoms)
    adj = [[] for _ in range(n_atoms)]
    for i, j in graph.bonds:
        adj[i].append(j)
        adj[j].append(i)

    # BFS to find path from init to edge
    queue = deque([init])
    visited = [False] * n_atoms
    parent = [None] * n_atoms
    visited[init] = True

    found = False
    while queue:
        current = queue.popleft()
        if current == edge:
            found = True
            break
        for neighbor in adj[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = current
                queue.append(neighbor)

    if not found:
        return []

    # Reconstruct path
    path = [edge]
    while path[-1] != init:
        path.append(parent[path[-1]])
    path.reverse()
    return path

class LammpsReacter:

    def __init__(self, typifier):
        self.typifier = typifier

    def react(self, name: str, a: mp.Struct, b: mp.Struct, workdir: Path, label: str = ""):
        # 只选择label匹配的link site、break、end
        alinks = [site for site in a["links"] if getattr(site, "label", "") == label]
        abreaks = [br for br in a["breaks"] if getattr(br, "label", "") == label]
        aends = [end for end in a["ends"] if getattr(end, "label", "") == label]
        blinks = [site for site in b["links"] if getattr(site, "label", "") == label]
        bbreaks = [br for br in b["breaks"] if getattr(br, "label", "") == label]
        bends = [end for end in b["ends"] if getattr(end, "label", "") == label]
        if not (alinks and blinks and aends and bends):
            raise ValueError(f"No matching link/end site with label '{label}' found in one or both fragments.")
        site_a = alinks[0]
        break_a = abreaks[0] if abreaks else None
        edge_a = aends[0].atom
        site_b = blinks[0]
        break_b = bbreaks[0] if bbreaks else None
        edge_b = bends[0].atom

        self.typifier.typify(a, workdir=workdir/a["name"])
        self.typifier.typify(b, workdir=workdir/b["name"])

        graph_a = a.get_topology()
        graph_b = b.get_topology()

        mask_a = _collect_atoms_index_between_to_site(graph_a, a.atoms.index(site_a.anchor), a.atoms.index(edge_a))
        mask_b = _collect_atoms_index_between_to_site(graph_b, b.atoms.index(site_b.anchor), b.atoms.index(edge_b))

        sub_a = a.get_substruct(mask_a)
        sub_b = b.get_substruct(mask_b)
        pre = mp.Struct.concat(name, [sub_a, sub_b])

        # Find anchor indices in the new structure
        ia = pre.atoms.index(site_a.anchor)
        ib = pre.atoms.index(site_b.anchor)
        delete_ids = [pre.atoms.index(atom) for atom in site_a.deletes + site_b.deletes]

        post = pre.__class__(name=name)
        # Copy all data from pre to post (deepcopy or custom logic may be needed)
        for key in pre:
            post[key] = pre[key].copy() if hasattr(pre[key], 'copy') else pre[key]

        post.def_bond(
            post.atoms[ia], post.atoms[ib],
        )
        post.del_atoms(site_a.deletes)
        post.del_atoms(site_b.deletes)

        if break_a:
            post.del_bond(break_a.this, break_a.that)
        if break_b:
            post.del_bond(break_b.this, break_b.that)

        self.typifier.typify(post, workdir=workdir/name)

        mp.io.write_lammps_molecule(workdir/ name / f"{name}_pre.mol", pre.to_frame())
        mp.io.write_lammps_molecule(workdir/ name / f"{name}_post.mol", post.to_frame())

        # create mapping file

        init_ids = [ia, ib]
        edge_atoms = [edge_a, edge_b]
        edge_ids = []

        for n in edge_atoms:
            try:
                idx = pre.atoms.index(n)
                edge_ids.append(idx)
            except Exception:
                pass

        with open(workdir / name / f'{name}.map', 'w') as f:
            f.write(f'# {name} mapping file\n\n')
            f.write(f'{len(pre["atoms"])} equivalences\n')
            f.write(f'{len(edge_ids)} edgeIDs\n')
            f.write(f'{len(delete_ids)} deleteIDs\n')
            f.write('\n')

            f.write('InitiatorIDs\n\n')
            for i in init_ids:
                f.write(f'{i+1}\n')
            f.write('\n')

            if edge_ids:
                f.write('EdgeIDs\n\n')
                for i in edge_ids:
                    f.write(f'{i+1}\n')
                f.write('\n')

            if delete_ids:
                f.write('DeleteIDs\n\n')
                for i in delete_ids:
                    f.write(f'{i+1}\n')
                f.write('\n')

            f.write('Equivalences\n\n')
            for idx, (i, j) in enumerate(zip(pre['atoms'], post['atoms'])):
                f.write(f'{idx+1} {idx+1}\n')
        return post