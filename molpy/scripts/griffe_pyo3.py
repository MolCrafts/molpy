"""Griffe extension: a PyO3 class's constructor signature, as Python sees it.

A ``#[pyclass]`` is constructed through its ``#[new]``, which PyO3 exposes as
the class's ``__text_signature__`` (so ``inspect.signature(molpy.conformer.
Conformer)`` is ``(speed='medium', add_hydrogens=True, seed=None)``) and not
as an ``__init__``. Griffe reads a class's parameters from ``__init__`` only,
so under ``force_inspection`` every PyO3 class had none: the reference page
showed no constructor signature, and each class docstring's ``Parameters``
section was reported as naming parameters "not in the function signature".

For each inspected class that has a text signature and no ``__init__`` of its
own, this adds an ``__init__`` carrying exactly that signature, in place of
the inherited ``__new__`` whose docstring is CPython's generic one.
"""

from __future__ import annotations

import inspect
from typing import Any

import griffe

_GENERIC_NEW = object.__new__.__doc__


class PyO3Constructors(griffe.Extension):
    def on_class_members(self, *, node: Any, cls: griffe.Class, **kwargs: Any) -> None:
        obj = getattr(node, "obj", None)
        if not isinstance(obj, type) or "__init__" in cls.members:
            return
        if not getattr(obj, "__text_signature__", None):
            return
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            return
        parameters = griffe.Parameters(
            griffe.Parameter("self", kind=griffe.ParameterKind.positional_or_keyword),
            *(
                griffe.Parameter(
                    p.name,
                    kind=getattr(griffe.ParameterKind, p.kind.name.lower()),
                    default=None if p.default is p.empty else repr(p.default),
                )
                for p in signature.parameters.values()
            ),
        )
        init = griffe.Function(
            "__init__", parameters=parameters, returns="None", parent=cls
        )
        new = cls.members.get("__new__")
        if (
            new is not None
            and new.docstring
            and new.docstring.value == inspect.cleandoc(_GENERIC_NEW)
        ):
            cls.del_member("__new__")
        cls.set_member("__init__", init)
