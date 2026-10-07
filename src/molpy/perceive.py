"""Chemical perception — :mod:`molrs.perceive`, mirrored by identity.

Every perception is a free function: ``perceive_<fact>`` returns a side table
(``perceive_rings(mol) -> RingInfo``) and ``assign_<fact>`` writes the fact
onto a copy (``assign_rings``, ``assign_aromaticity``, ``assign_stereo``,
``assign_bond_orders``, ``assign_kekule_bond_orders``,
``assign_rotatable_bonds``, ``assign_equivalence_classes``,
``assign_bcc_bond_types``, …); ``add_hydrogens`` is an edit. SMARTS matching
and reactions (``SmartsPattern``, including
``SmartsPattern.from_environment``; ``SmartsMatch``, ``Reaction``) and
coarse-grained bead-pattern matching (``SubgraphMatcher``) are here too;
``mp.perceive.assign_rings is molrs.perceive.assign_rings``.
"""

from molrs.perceive import *  # noqa: F403
from molrs.perceive import __all__ as __all__
