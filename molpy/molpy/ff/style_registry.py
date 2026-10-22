"""The style registry (``molrs::ff::style_registry``): register a new style
or a new category of the force-field IR (:mod:`molpy.ff.ir`) from Python,
with nothing in molrs rebuilt.

The force-field IR adopts the LAMMPS standard; its *shape* is data. A
**category** names how many atoms a term has and which Frame block its terms
live in; a **style** names its ordered parameters, each with a dimension, and
its energy — as an expression, a vectorised Python kernel, or both. Anything
of that form registers into the process-wide registry every
:class:`~molpy.ff.compile.PotentialCompiler` reads, and then prices at both compile
doors and in MD exactly like a built-in. What does not conform is refused
with an :class:`~molpy.ff.ir.IrError` subclass naming the offending item.

A new bond style, LAMMPS ``bond_style fene``, by its expression::

    from molpy.ff import ir, style_registry

    class Fene(style_registry.StyleDeclaration):
        category = "bond"
        name = "fene"
        params = [ir.ParamSpec("k", "E/L^2"), ir.ParamSpec("r0", "L"),
                  ir.ParamSpec("epsilon", "E"), ir.ParamSpec("sigma", "L")]
        expression = ("-0.5*k*r0^2*log(1-(r/r0)^2)"
                      "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)")

    ff.def_style("bond", "fene").def_type("B-B", b, b, k=30.0, r0=1.5,
                                          epsilon=1.0, sigma=1.0)

The same style priced by numpy instead (one call per style per evaluation;
``q`` is the bond length of every term, each parameter an array beside it)::

    def fene(r, k, r0, epsilon, sigma):
        ...
        return e, de_dr

    style_registry.register_style("bond", "fene/np", params=Fene.params, kernel=fene)

Calling conventions (every tier): ``q`` is the category's coordinate — ``r``
(length), ``theta`` in [0, π] or ``phi`` in (−π, π] (radians); parameters
arrive **as stored** (angle values in degrees, the expression converts with
``(pi/180)`` and defines ``pi=3.141592653589793`` last, Lepton having no named
constants); a kernel returns the **unweighted** energy per term
and its derivative (or ``∂E/∂x`` for a compound kernel, not the force). The
pair weight, the cutoff and the chain rule onto Cartesian forces are the
form kernels'.

Refusals are the :class:`~molpy.ff.ir.IrError` subclasses of
:mod:`molpy.ff.ir`. A Python kernel that raises during an evaluation, a
compile or a registration surfaces as
:class:`~molpy.ff.ir.KernelShapeError` with the original exception as
``__cause__``.
"""

from molpy._core import style_registry as _style_registry

from molpy.ff.style_declaration import StyleDeclaration

register_category = _style_registry.register_category

register_style = _style_registry.register_style

register_engine_form = _style_registry.register_engine_form

unregister_style = _style_registry.unregister_style

styles = _style_registry.styles

categories = _style_registry.categories

evaluate = _style_registry.evaluate

StyleDeclaration.__module__ = __name__

__all__ = [
    "StyleDeclaration",
    "categories",
    "evaluate",
    "register_category",
    "register_engine_form",
    "register_style",
    "styles",
    "unregister_style",
]
