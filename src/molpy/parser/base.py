from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from lark import Lark, Tree

T = TypeVar("T")  # Output IR type


@dataclass(slots=True)
class GrammarConfig:
    """
    Configuration for the grammar-backed parser.
    """

    grammar_path: Path  # Path to a .lark grammar file
    start: str  # Start rule name in grammar
    parser: Literal["lalr", "earley"] = "lalr"
    propagate_positions: bool = False  # Lark option
    maybe_placeholders: bool = False  # Lark option
    auto_reload: bool = True  # Reload grammar on file change


class GrammarParserBase[T](ABC):
    """
    Base class for parsers backed by an external Lark grammar file.

    Lifecycle:
      - Construct with a GrammarConfig
      - Call parse_tree(text) to get a Lark Tree
      - Implement build(tree) in subclasses to map Tree -> IR
      - Call parse(text) to get your IR

    Features:
      - Grammar is compiled once and cached
      - If auto_reload=True, grammar file mtime is checked before each parse
    """

    def __init__(self, config: GrammarConfig):
        self.config = config
        self._lark: Lark | None = None
        self._mtime: float | None = None
        self._compile_grammar(force=True)

    # ---------- Public API ----------

    def parse_tree(self, text: str) -> Tree:
        """
        Parse input string into a Lark parse tree.
        """
        self._maybe_reload()
        assert self._lark is not None
        return self._lark.parse(text)

    # ---------- Internal helpers ----------

    def _maybe_reload(self) -> None:
        if not self.config.auto_reload:
            return
        try:
            mtime = self.config.grammar_path.stat().st_mtime
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Grammar file not found: {self.config.grammar_path}"
            ) from e

        if self._mtime is None or mtime != self._mtime:
            self._compile_grammar(force=True)

    def _compile_grammar(self, *, force: bool = False) -> None:
        if self._lark is not None and not force:
            return

        path = self.config.grammar_path
        if not path.exists():
            raise FileNotFoundError(f"Grammar file not found: {path}")

        grammar_text = path.read_text(encoding="utf-8")

        self._lark = Lark(
            grammar_text,
            start=self.config.start,
            parser=self.config.parser,
            propagate_positions=self.config.propagate_positions,
            maybe_placeholders=self.config.maybe_placeholders,
            import_paths=[str(path.parent)],
            keep_all_tokens=True,
        )
        self._mtime = path.stat().st_mtime
