import molpy as mp
import numpy as np

class TestPairPotentialBase:

    def test_or_frame(self):
        
        pair_pot = mp.potential.pair.PairPotential()

        frame = mp.Frame("atoms", "bonds", "angles", "pairs")
        frame["atoms"]["x"] = [0.0000, 0.75695, -0.75695]
        frame["atoms"]["y"] = [-0.06556, 0.52032, 0.52032]
        frame["atoms"]["z"] = [0.000, 0.00000, 0.00000]
        frame["atoms"]["charge"] = [0, 1, 1]
        frame["pairs"]["i"] = [0, 0, 1]
        frame["pairs"]["j"] = [1, 2, 2]
        frame["pairs"]["type"] = [1, 1, 2]

        self = pair_pot
        pair_pot.or_frame(lambda r, idx, types: None)(self, frame)
        pair_pot.or_frame('charge')(lambda r, idx, types, charges: None)(self, frame)
        # assert False