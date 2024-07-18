from pathlib import Path
import molpy as mp

class TestAmber:

    def test_tfsi(self, test_data_path):
        work_dir = Path("tests/tmp/param")
        gaff_parameterizer = mp.builder.parameterizer.AmberTool(work_dir)
        raw_tfsi = mp.io.load_struct(test_data_path / "data/pdb/tfsi.pdb")
        gaff_parameterizer.parameterize(raw_tfsi, {"-c": "bcc", "-n": "-1"}, )

        raw_li = mp.io.load_struct(test_data_path / "data/pdb/li+.pdb")

        li_ff = mp.ForceField('Li', 'real')
        astyle = li_ff.def_atomstyle('full')
        astyle.def_atomtype('Li', 1, **{mp.Alias.charge: 0.75, mp.Alias.mass: 6.941})
        pstyle = li_ff.def_pairstyle('lj/cut/coul/long')
        pstyle.def_pairtype('Li', 1, 1, 2.02590e-01, 7.65672e-02)

        tfsi = li_parameterizer = mp.builder.parameterizer.Custom(li_ff, work_dir)
        li = li_parameterizer.parameterize(raw_li)

        # forcefield = parameterize.load_forcefield(format='lammps')
        # tfsi = gaff_parameterizer.load_struct(tfsi.name, format='pdb')
        # li = li_parameterizer.load_struct(li.name, format='pdb')
        
        assert li.n_atoms == 1
        assert li.atoms.charge[0] == 0.75

        # import molpack as mpk

        # app = mpk.Molpack(work_dir = "tests/tmp/pack")
        # region = mp.region.Cube([0, 0, 0], 10)
        # app.add_struct(tfsi, 1, region)
        # app.add_struct(li, 1, region)
        # app.set_optimizer()
        # app.optimize()

        # structs = app.get_structs()
        # frame = mp.Frame()
        # frame.box = app.get_box()
        # for struct in structs:
        #     if struct.name.startswith("tfsi"):
        #         new = tfsi.clone()
        #         new.xyz = struct.xyz
        #     elif struct.name.startswith("li"):
        #         new = li.clone()
        #         new.xyz = struct.xyz
        #     new.id += frame.n_atoms
        #     new.molid += 1
        #     frame.add_struct(new)

        # frame.save("tests/tmp/tfsi_li.data", format="LAMMPS Data")


        # assert tfsi.n_atoms == 15
        # assert li.n_atoms == 1