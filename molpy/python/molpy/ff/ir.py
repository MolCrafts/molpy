"""The force-field IR as vocabulary (``molrs::ff::ir``): what a category, a
style and its parameters are.

The force-field IR adopts the LAMMPS standard; its *shape* is data. A
**category** (:class:`CategorySpec`) names how many atoms a term has and
which Frame block its terms live in; a **style** (:class:`StyleSpec`) names
its ordered parameters (:class:`ParamSpec`), each with a dimension, and its
energy. Registering a new category or style — by expression, by a vectorised
Python kernel, or both — is :mod:`molpy.ff.style_registry`'s.

Refusals are :class:`IrError` (a ``ValueError``) subclasses named after the
Rust variants with an ``Error`` suffix (``IrError::Dimension`` is
:class:`DimensionError`): :class:`SealedError`, :class:`ConflictError`,
:class:`UnboundVariableError`, :class:`UnknownFunctionError`, :class:`KernelShapeError`,
:class:`NoKernelError`, :class:`MissingParamError`, :class:`BadValueError`,
:class:`DerivativeError`, … A Python
kernel that raises during an evaluation, a compile or a registration surfaces
as :class:`KernelShapeError` with the original exception as ``__cause__``.
"""

from molpy import _core

IrError = _core.ir.IrError

UnknownCategoryError = _core.ir.UnknownCategoryError

BadNameError = _core.ir.BadNameError

ArityError = _core.ir.ArityError

BlockNameError = _core.ir.BlockNameError

ReservedParamError = _core.ir.ReservedParamError

DuplicateParamError = _core.ir.DuplicateParamError

DimensionError = _core.ir.DimensionError

ParseError = _core.ir.ParseError

UnboundVariableError = _core.ir.UnboundVariableError

UnknownFunctionError = _core.ir.UnknownFunctionError

FunctionArityError = _core.ir.FunctionArityError

PointError = _core.ir.PointError

CoordinateMismatchError = _core.ir.CoordinateMismatchError

DerivativeError = _core.ir.DerivativeError

DisagreeError = _core.ir.DisagreeError

AsymmetricError = _core.ir.AsymmetricError

SealedError = _core.ir.SealedError

ConflictError = _core.ir.ConflictError

NoKernelError = _core.ir.NoKernelError

NoMixingError = _core.ir.NoMixingError

MissingParamError = _core.ir.MissingParamError

BadValueError = _core.ir.BadValueError

KernelShapeError = _core.ir.KernelShapeError

NoEngineFormError = _core.ir.NoEngineFormError

FormConflictError = _core.ir.FormConflictError

NoFormError = _core.ir.NoFormError

OutOfImageError = _core.ir.OutOfImageError

MalformedError = _core.ir.MalformedError

ParamSpec = _core.ir.ParamSpec

StyleSpec = _core.ir.StyleSpec

CategorySpec = _core.ir.CategorySpec

__all__ = [
    "ArityError",
    "AsymmetricError",
    "BadNameError",
    "BadValueError",
    "BlockNameError",
    "CategorySpec",
    "ConflictError",
    "CoordinateMismatchError",
    "DerivativeError",
    "DimensionError",
    "DisagreeError",
    "DuplicateParamError",
    "FormConflictError",
    "FunctionArityError",
    "IrError",
    "KernelShapeError",
    "MalformedError",
    "MissingParamError",
    "NoEngineFormError",
    "NoFormError",
    "NoKernelError",
    "NoMixingError",
    "OutOfImageError",
    "ParamSpec",
    "ParseError",
    "PointError",
    "ReservedParamError",
    "SealedError",
    "StyleSpec",
    "UnboundVariableError",
    "UnknownCategoryError",
    "UnknownFunctionError",
]
