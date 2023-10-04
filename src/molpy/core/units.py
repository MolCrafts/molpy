import pint
from typing import Any

__all__ = ["convert"]

ureg = pint.UnitRegistry()


def convert(value: Any, src_unit: str, target_unit: str):
    return ureg.Quantity(value, src_unit).to(target_unit).magnitude
