from pathlib import Path
import molpy as mp


class LammpsReacter:

    def __init__(self, typifier):
        self.typifier = typifier

    def react(self, monomer, workdir: Path):

        rxn_name = monomer["name"]
        pre = monomer
        post = monomer.copy()

        for port in post["ports"]:

            del_atoms = port.delete
            post.del_atoms(del_atoms)
            post.add_atom(port.that)
            post.add_bond(port.this, port.that)
    
        if self.typifier:
            self.typifier.typify(post)

        mp.io.write_lammps_molecule(monomer, workdir/f"pre_{rxn_name}.mol")
        mp.io.write_lammps_molecule(post, workdir/f"post_{rxn_name}.mol")

        # create mapping filef
