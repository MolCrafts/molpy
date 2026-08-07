"""Antechamber (AC) file reader (molrs-backed)."""

from pathlib import Path

from molpy.core.fields import CHARGE, FieldFormatter
from molrs import Frame

from .base import DataReader


class AcFieldFormatter(FieldFormatter):
    """Antechamber .ac field name translation."""

    _field_formatters = {
        "q": CHARGE,
    }


class AcReader(DataReader):
    """Reader for Antechamber .ac format files (via molrs)."""

    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self._file = Path(file)

    def read(self, frame: Frame | None = None) -> Frame:
        """Read .ac file into a Frame."""
        import molrs.io

        del frame
        if not self._file.is_file():
            raise FileNotFoundError(self._file)
        try:
            loaded = molrs.io.read_ac(str(self._file))
        except OSError as exc:
            raise ValueError(f"Failed to read AC file: {exc}") from exc
        # Canonicalize charge if present as q
        self._formatter.canonicalize_frame(loaded)
        return loaded

    _formatter = AcFieldFormatter()
