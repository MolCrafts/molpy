"""molpy.op — the native numeric base, a verbatim re-export of the native ``op`` module.

Weighted superposition (``superpose``, ``Fit``) and centroids (``centroid``)
live in the native core. Users spell everything ``molpy.op.<Name>``; the
objects are identical to their ``op`` counterparts.
"""

from molrs.op import *  # noqa: F403
from molrs.op import __all__ as __all__
