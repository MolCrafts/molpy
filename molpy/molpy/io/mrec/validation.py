"""Runtime validation of the mrec record schema.

The checks live in ``molrs::io::mrec::validation``; this module is their Python
name, not a second implementation.
"""

from molpy._core import mrec as _mrec

validate_frame = _mrec.validation.validate_frame

__all__ = [
    "validate_frame",
]
