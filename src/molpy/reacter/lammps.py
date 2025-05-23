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

    def react(self, name: str, pre: mp.Struct, init_a, init_b, edge_a, edge_b, deletes=[], workdir: Path=Path.cwd()):

        system = self.typifier.get_forcefield(pre, workdir=workdir/pre["name"])

        init_id_a = pre.atoms.index(init_a)
        init_id_b = pre.atoms.index(init_b)
        edge_id_a = pre.atoms.index(edge_a)
        edge_id_b = pre.atoms.index(edge_b)
        delete_ids = [pre.atoms.index(atom) for atom in deletes]

        mp.io.write_lammps_molecule(workdir / name / f"{name}_pre.mol", pre.to_frame())

        post = pre(name=name)

        post.def_bond(
            post.atoms[init_id_a], post.atoms[init_id_b],
        )
        if deletes:
            post.del_bonds(deletes)

        system = self.typifier.get_forcefield(post, workdir=workdir/name, system=system)

        mp.io.write_lammps_molecule(workdir/ name / f"{name}_post.mol", post.to_frame())
        mp.io.write_lammps_forcefield(workdir / name / f"{name}.ff", system)

        init_ids = [init_id_a, init_id_b]
        edge_ids = [edge_id_a, edge_id_b]

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