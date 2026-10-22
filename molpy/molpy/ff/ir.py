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

from molpy._core import ir as _ir

IrError = _ir.IrError

UnknownCategoryError = _ir.UnknownCategoryError

BadNameError = _ir.BadNameError

ArityError = _ir.ArityError

BlockNameError = _ir.BlockNameError

ReservedParamError = _ir.ReservedParamError

DuplicateParamError = _ir.DuplicateParamError

DimensionError = _ir.DimensionError

ParseError = _ir.ParseError

UnboundVariableError = _ir.UnboundVariableError

UnknownFunctionError = _ir.UnknownFunctionError

FunctionArityError = _ir.FunctionArityError

PointError = _ir.PointError

CoordinateMismatchError = _ir.CoordinateMismatchError

DerivativeError = _ir.DerivativeError

DisagreeError = _ir.DisagreeError

AsymmetricError = _ir.AsymmetricError

SealedError = _ir.SealedError

ConflictError = _ir.ConflictError

NoKernelError = _ir.NoKernelError

NoMixingError = _ir.NoMixingError

MissingParamError = _ir.MissingParamError

BadValueError = _ir.BadValueError

KernelShapeError = _ir.KernelShapeError

NoEngineFormError = _ir.NoEngineFormError

FormConflictError = _ir.FormConflictError

NoFormError = _ir.NoFormError

OutOfImageError = _ir.OutOfImageError

MalformedError = _ir.MalformedError

ParamSpec = _ir.ParamSpec

StyleSpec = _ir.StyleSpec

CategorySpec = _ir.CategorySpec

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
