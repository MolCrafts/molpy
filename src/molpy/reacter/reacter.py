import molpy as mp
import numpy as np

def lammps_reacter(rxn_name, pre, initiator_names, edge_names=[], delete_names=[], type_maps={}):

    for i, atom in enumerate(pre['atoms']):
        atom['id'] = i

    mp.io.write_lammps_molecule(pre, f'pre_{rxn_name}.mol')

    post = pre.copy()
    for del_ in delete_names:
        post.unlink_(del_)
    for name, type_ in type_maps.items():
        atom = post['atoms'].get_by(lambda atom: atom['name'] == name)
        atom['type'] = type_
        for bond in post['bonds']:
            if atom in bond:
                bond['type'] = f"{bond.itom['type']}-{bond.jtom['type']}"
        for angle in post['angles']:
            if atom in angle:
                angle['type'] = f"{angle.itom['type']}-{angle.jtom['type']}-{angle.ktom['type']}"
        for dihedral in post['dihedrals']:
            if atom in dihedral:
                dihedral['type'] = f"{dihedral.itom['type']}-{dihedral.jtom['type']}-{dihedral.ktom['type']}-{dihedral.ltom['type']}"
    post = post.link_(*initiator_names)

    mp.io.write_lammps_molecule(post, f'post_{rxn_name}.mol')
    init_ids = [pre['atoms'].get_by(lambda atom: atom['name'] == name)['id'] for name in initiator_names]
    edge_ids = [pre['atoms'].get_by(lambda atom: atom['name'] == name)['id'] for name in edge_names]
    delete_ids = [post['atoms'].get_by(lambda atom: atom['name'] == name)['id'] for name in delete_names]
    with open(f'{rxn_name}.map', 'w') as f:

        f.write(f'# {name} mapping file\n')
        f.write(f'\n')
        f.write(f'{len(pre['atoms'])} equivalences\n')
        if edge_names:
            f.write(f'{len(edge_names)} edgeIDs\n')
        if delete_names:
            f.write(f'{len(delete_names)} deleteIDs\n')

        f.write(f'\n')

        f.write(f'InitiatorIDs\n\n')
        for i in init_ids:
            f.write(f'{i+1}\n')

        f.write(f'\n')

        f.write(f'EdgeIDs\n\n')
        for i in edge_ids:
            f.write(f'{i+1}\n')

        f.write(f'\n')
        if delete_names:
            f.write(f'DeleteIDs\n\n')
            for i in delete_ids:
                f.write(f'{i+1}\n')
            f.write(f'\n')

        f.write('Equivalences\n\n')
        for i, j in zip(pre['atoms'], post['atoms']):
            f.write(f'{i["id"]+1} {j["id"]+1}\n')

    return post

# def link_(self, other: "Struct", initiator_names: tuple[str, str], delete_names: list[str] = [], type_maps: dict[str, str] = {}):

#     from_ = self.get_atom_by_name(initiator_names[0])
#     to_ = other.get_atom_by_name(initiator_names[1])

#     other.move_(np.array([from_["x"], from_["y"], from_["z"]]) - np.array([to_["x"], to_["y"], to_["z"]]))
#     # rotate

#     for delete_name in delete_names:
#         self.del_atom_(delete_name)

#     for name, type_ in type_maps.items():
#         atom = self.get_atom_by_name(name)
#         atom["type"] = type_
#         for bond in self["bonds"]:
#             if atom in bond:
#                 bond["type"] = f"{bond.itom['type']}-{bond.jtom['type']}"
#         for angle in self["angles"]:
#             if atom in angle:
#                 angle["type"] = f"{angle.itom['type']}-{angle.jtom['type']}-{angle.ktom['type']}"
#         for dihedral in self["dihedrals"]:
#             if atom in dihedral:
#                 dihedral["type"] = f"{dihedral.itom['type']}-{dihedral.jtom['type']}-{dihedral.ktom['type']}-{dihedral.ltom['type']}"


#     from_atom = self.get_atom_by_name(from_)
#     to_atom = self.get_atom_by_name(to_)

#     if from_atom is None or to_atom is None:
#         raise ValueError("Atom not found")
#     if from_atom == to_atom:
#         raise ValueError("Cannot link atom to itself")
#     if from_atom > to_atom:  # i-j -> from-to
#         i, j = to_atom, from_atom
#     else:
#         i, j = from_atom, to_atom
#     self["bonds"].append(
#         Bond(
#             i,
#             j,
#             type=f"{"-".join([atom["type"] for atom in [from_atom, to_atom]])}",
#         )
#     )

#     # add angle
#     from_atom_bonds = self.get_bonds_by_atom(from_atom)
#     from_atom_neighbors = [
#         bond.itom if bond.jtom == from_atom else bond.jtom
#         for bond in from_atom_bonds
#     ]
#     for neighbor in from_atom_neighbors:  # i-j-k
#         if neighbor == to_atom:
#             continue
#         if neighbor < to_atom:
#             i, j, k = neighbor, from_atom, to_atom
#         else:
#             i, j, k = to_atom, from_atom, neighbor
#         new_angle = Angle(
#             i,
#             j,
#             k,
#             type=f"{"-".join([i["type"], j["type"], k["type"]])}",
#         )
#         self["angles"].append(new_angle)
#     to_atom_bonds = self.get_bonds_by_atom(to_atom)
#     to_atom_neighbors = [
#         bond.itom if bond.jtom == to_atom else bond.jtom for bond in to_atom_bonds
#     ]
#     for neighbor in to_atom_neighbors:
#         if neighbor == from_atom:
#             continue
#         if neighbor < from_atom:
#             i, j, k = neighbor, to_atom, from_atom
#         else:
#             i, j, k = from_atom, to_atom, neighbor
#         new_angle = Angle(
#             i,
#             j,
#             k,
#             type=f"{"-".join([i["type"], j["type"], k["type"]])}",
#         )
#         print(f"Adding angle {i['id']} {j['id']} {k['id']}")
#         self["angles"].append(new_angle)

#     # add dihedral
#     for i, l in product(from_atom_neighbors, to_atom_neighbors):
#         if i == l or i == to_atom or l == from_atom:  # loop
#             continue
#         if from_atom < to_atom:
#             i, j, k, l = i, from_atom, to_atom, l
#         else:
#             i, j, k, l = l, to_atom, from_atom, i
#         new_dihedral = Dihedral(
#             i,
#             j,
#             k,
#             l,
#             type=f"{'-'.join([
#             i["type"],
#             j["type"],
#             k["type"],
#             j["type"],
#         ])}",
#         )
#         print(f"Adding dihedral {i['id']} {j['id']} {k['id']} {l['id']}")
#         self["dihedrals"].append(new_dihedral)

#     return self