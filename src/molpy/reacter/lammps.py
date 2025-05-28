from pathlib import Path
import molpy as mp
from collections import deque
from molpy.core import Topology


def find_path_with_valid_branches(g: mp.Topology, start: int, end: int):
    """
    Get the main chain (shortest path) from start to end and all valid branch atom indices attached to the main chain.

    Branch definition:
      - For each neighbor of every atom on the main chain (shortest path),
      - If the neighbor is not on the main chain and has not been visited,
      - Start a DFS from this neighbor, and stop expanding when encountering any main chain atom (do not traverse back to the main chain or across to the other side).

    Args:
        g (mp.Topology): Molecular topology graph (subclass of igraph Graph).
        start (int): Index of the starting atom of the main chain.
        end (int): Index of the ending atom of the main chain.

    Returns:
        set: Set of indices including the main chain (excluding start) and all valid branch atoms.
    """
    # Get the main chain (shortest path), including start and end
    path = g.get_shortest_paths(start, to=end, output="vpath")[0]
    if not path or len(path) < 2:
        raise ValueError("Path must include at least two nodes")

    path_set = set(path)
    visited = set([start])  # Track visited nodes to avoid revisiting
    result = set(path)  # Result set, main chain (excluding start)

    # Traverse the main chain (from path[1], skipping start)
    for i in range(1, len(path)):
        node = path[i]
        visited.add(node)
        # Check each neighbor of the main chain node
        for nbr in g.neighbors(node):
            if nbr in visited or nbr in path_set:
                continue  # Skip already visited and main chain nodes
            # Start DFS from this neighbor, stop at main chain nodes
            stack = [nbr]
            branch_visited = set()
            while stack:
                curr = stack.pop()
                if curr in visited or curr in path_set:
                    continue  # Do not traverse back to main chain or visited nodes
                branch_visited.add(curr)
                visited.add(curr)
                result.add(curr)
                # Expand the branch
                for next_nbr in g.neighbors(curr):
                    if next_nbr not in visited and next_nbr not in path_set:
                        stack.append(next_nbr)
    return result


class LammpsReacter:

    def __init__(self, typifier):
        self.typifier = typifier

    def react(
        self,
        name: str,
        pre: mp.Struct,
        init_a,
        init_b,
        edge_a,
        edge_b,
        deletes=[],
        workdir: Path = Path.cwd(),
    ):

        (workdir / name).mkdir(parents=True, exist_ok=True)
        init_id_a = pre.atoms.index(init_a)
        init_id_b = pre.atoms.index(init_b)
        edge_id_a = pre.atoms.index(edge_a)
        edge_id_b = pre.atoms.index(edge_b)
        delete_ids = [pre.atoms.index(atom) for atom in deletes]

        mp.io.write_lammps_molecule(
            workdir / name / f"{pre['name']}.mol", pre.to_frame()
        )

        post = pre(name=name)

        post.def_bond(
            post.atoms[init_id_a],
            post.atoms[init_id_b],
        )
        if deletes:
            post.del_bonds(deletes)

        frame = self.typifier.get_forcefield(post, workdir=workdir / name)

        mp.io.write_lammps_molecule(workdir / name / f"{name}.mol", post.to_frame())
        mp.io.write_lammps_forcefield(workdir / name / f"{name}.ff", frame.forcefield)

        init_ids = [init_id_a, init_id_b]
        edge_ids = [edge_id_a, edge_id_b]

        with open(workdir / name / f"{name}.map", "w") as f:
            f.write(f"# {name} mapping file\n\n")
            f.write(f'{len(pre["atoms"])} equivalences\n')
            f.write(f"{len(edge_ids)} edgeIDs\n")
            f.write(f"{len(delete_ids)} deleteIDs\n")
            f.write("\n")

            f.write("InitiatorIDs\n\n")
            for i in init_ids:
                f.write(f"{i+1}\n")
            f.write("\n")

            if edge_ids:
                f.write("EdgeIDs\n\n")
                for i in edge_ids:
                    f.write(f"{i+1}\n")
                f.write("\n")

            if delete_ids:
                f.write("DeleteIDs\n\n")
                for i in delete_ids:
                    f.write(f"{i+1}\n")
                f.write("\n")

            f.write("Equivalences\n\n")
            for idx, (i, j) in enumerate(zip(pre["atoms"], post["atoms"])):
                f.write(f"{idx+1} {idx+1}\n")
        return post

    def find_template(self, name: str, struct: mp.Struct, end1, end2):
        """
        Find a template for the reaction based on the end atoms.
        This function assumes that the structure is connected and has a valid topology.
        """
        graph = struct.get_topology()
        end1_id = struct.atoms.index(end1)
        end2_id = struct.atoms.index(end2)

        # Collect all atoms between end1 and end2
        path_atoms = find_path_with_valid_branches(graph, end1_id, end2_id)

        # Create a new structure with the collected atoms
        template = struct.get_substruct(name, list(path_atoms))
        return template
