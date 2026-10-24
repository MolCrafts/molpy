"""``StyleDeclaration``, the class form of
:func:`molpy.ff.style_registry.register_style`.

Private: :mod:`molpy.ff.style_registry` is its public path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from typing import Any, ClassVar, Literal

from molpy import _core


class StyleDeclaration:
    """Declare a force-field IR style as a class; defining the subclass
    registers it (:func:`register_style`).

    Attributes
    ----------
    category : str
        The category (built-in, or from :func:`register_category`).
    name : str
        The style name.
    params : list of ParamSpec, or dict of {name: dim}
        The per-type parameters, in order.
    style_params : list of ParamSpec, or dict of {name: dim}, optional
        Style-level parameters (``cutoff``, ``mixing``, ``special`` keep their
        reserved meanings).
    expression : str, optional
        The energy as an expression.
    special : {"lj", "coul"}, optional
        A pair style's special-bonds class.
    samples : list of dict, optional
        Registration samples, ``{"q": (lo, hi), <param>: value}``.
    compound : bool
        ``kernel`` takes positions ``x`` (``(n, arity, 3)``), not ``q``.
    replace : bool
        Replace a style of the name registered at run time.
    lammps : {"positional", "positional:<name>"}, optional
        The style's LAMMPS form (:func:`register_style`'s ``lammps``).

    A ``kernel(self, q, **params) -> (e, de_dq)`` method (or
    ``kernel(self, x, **params) -> (e, grad)`` for a compound style) prices
    the style by Python; with an ``expression`` beside it the two must agree.
    Pass ``register=False`` in the class statement for an intermediate base
    that declares no style.

    Examples
    --------
    >>> import numpy as np
    >>> from molpy.ff import style_registry
    >>> class Quartic(style_registry.StyleDeclaration):
    ...     category = "bond"
    ...     name = "quartic/doc"
    ...     params = {"k": "E/L^4", "r0": "L"}
    ...     expression = "k*(r-r0)^4"
    ...     def kernel(self, r, k, r0):
    ...         d = r - r0
    ...         return k * d**4, 4 * k * d**3
    >>> e, de = Quartic.evaluate([1.5], k=2.0, r0=1.0)
    >>> float(e[0]), float(de[0])
    (0.125, 1.0)
    >>> Quartic.unregister()
    """

    category: ClassVar[str]
    name: ClassVar[str]
    params: ClassVar[Sequence[Any] | Mapping[str, str]] = ()
    style_params: ClassVar[Sequence[Any] | Mapping[str, str]] = ()
    expression: ClassVar[str | None] = None
    special: ClassVar[Literal["lj", "coul"] | None] = None
    samples: ClassVar[Sequence[Mapping[str, Any]] | None] = None
    compound: ClassVar[bool] = False
    replace: ClassVar[bool] = False
    lammps: ClassVar[str | None] = None

    def __init_subclass__(cls, *, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not register:
            return
        missing = [
            a
            for a in ("category", "name")
            if not isinstance(getattr(cls, a, None), str)
        ]
        if missing:
            raise TypeError(
                f"{cls.__qualname__}: a StyleDeclaration declares {' and '.join(missing)} (a str)"
            )
        kernel: Callable[..., Any] | None = None
        if callable(getattr(cls, "kernel", None)):
            # `kernel` is optional, so it is not declared on the base class.
            kernel = getattr(cls(), "kernel")
        _core.style_registry.register_style(
            cls.category,
            cls.name,
            params=cls.params,
            style_params=cls.style_params,
            expression=cls.expression,
            kernel=kernel,
            compound=cls.compound,
            special=cls.special,
            samples=cls.samples,
            replace=cls.replace,
            lammps=cls.lammps,
        )

    @classmethod
    def evaluate(
        cls, q: Any = None, *, x: Any = None, **params: Any
    ) -> tuple[Any, Any]:
        """:func:`evaluate` of this style."""
        return _core.style_registry.evaluate(cls.category, cls.name, q, x=x, **params)

    @classmethod
    def unregister(cls) -> None:
        """:func:`unregister_style` this style."""
        _core.style_registry.unregister_style(cls.category, cls.name)
