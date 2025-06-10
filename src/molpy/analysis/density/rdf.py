from ..base import Compute
import freud


class RDF(Compute):

    def __init__(self, bins, r_max, r_min: int = 0, normalization_mode="exact"):
        super().__init__(
            kernel=freud.density.RDF(
                bins=bins,
                r_max=r_max,
                r_min=r_min,
                normalization_mode=normalization_mode,
            )
        )

    def compute(self, frame, reset: bool = False):

        box = frame.box
        points = frame["atoms"][["x", "y", "z"]].to_numpy()
        if hasattr(frame, "nblist"):
            nblist = frame.nblist
        else:
            nblist = None

        if nblist is None:
            self._kernel.compute(system=(box, points), reset=reset)
        else:
            self._kernel.compute(system=nblist, reset=reset)

        return self._kernel.rdf