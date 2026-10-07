"""The force-field style registry — :mod:`molrs.ff.style_registry`, by identity.

Register a new style (:func:`register_style`, or a :class:`StyleDeclaration`
subclass) or a new category (:func:`register_category`) from Python, with
nothing rebuilt; :func:`styles` / :func:`categories` describe what is
registered as ``mp.ff.ir.StyleSpec`` / ``mp.ff.ir.CategorySpec`` records;
:func:`evaluate` prices one registered style at sample points;
:func:`unregister_style` takes a style back. Refusals are ``mp.ff.ir.IrError``
subclasses. ``mp.ff.style_registry.X is molrs.ff.style_registry.X`` for every
name here.
"""

from molrs.ff.style_registry import *  # noqa: F403
from molrs.ff.style_registry import __all__ as __all__
