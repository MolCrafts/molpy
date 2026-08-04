"""Identity exports for the molrs-owned live graph view layer.

Molrs exposes NodeRef / RelationRef / Refs only. MolPy keeps Entity / Link /
Entities as **local** domain names (single definition site here).
"""

from molrs.views import GraphViews, NodeRef, Refs, RelationRef, _GraphViews

# Domain vocabulary — not dual APIs on molrs.
Entity = NodeRef
Link = RelationRef
Entities = Refs

__all__ = [
    "Entities",
    "Entity",
    "GraphViews",
    "Link",
    "NodeRef",
    "Refs",
    "RelationRef",
    "_GraphViews",
]
