//! Byte-level scanner for SMILES / SMARTS / `CGsmiles` strings.
//!
//! All three notations are pure ASCII, so byte indexing is safe and efficient.

use crate::line_notation::ast::Span;
use crate::line_notation::error::{Notation, SmilesError, SmilesErrorKind};

/// Zero-allocation cursor over a SMILES/SMARTS/`CGsmiles` input string.
pub(crate) struct Scanner<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    /// Byte offset the cursor treats as end of input.
    ///
    /// `bytes.len()` unless [`Scanner::set_limit`] narrowed it. Every read
    /// stops here, so a parser reading one slice of a larger string cannot
    /// wander out of it while still reporting absolute positions.
    limit: usize,
    notation: Notation,
}

impl<'a> Scanner<'a> {
    /// Create a new scanner over the whole of `input`, reading it as
    /// `notation`.
    ///
    /// The notation is stamped on every error [`Scanner::error`] and
    /// [`Scanner::error_at`] build, so a caller states it once here instead of
    /// at each diagnostic. It is also why a re-read of part of the input
    /// rewinds this scanner with [`Scanner::seek`] rather than constructing a
    /// second scanner over a substring: a substring scanner starts at position
    /// 0 and would report slice-relative positions.
    pub fn new(input: &'a str, notation: Notation) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
            limit: input.len(),
            notation,
        }
    }

    /// Treat byte offset `end` as end of input, clamped to the real end.
    ///
    /// A `CGsmiles` string is a sequence of `{…}` blocks and each block a
    /// sequence of `#NAME=body` entries, so several parsers read *slices* of
    /// one input while reporting absolute positions — which is why they seek
    /// into the whole string rather than scanning a substring. Seeking alone
    /// bounds only where a scan starts: a token left open at the slice's end
    /// (`{[#A]}.{#A=[#B;k=1}.{#B=[$]C}`, whose `[` is never closed) would run
    /// on into the next block and report a span covering text it has no
    /// business reading. Narrowing the limit makes the slice's end behave
    /// exactly like end of input, so such a token is refused where it was
    /// written.
    ///
    /// [`Scanner::seek`] is deliberately *not* bounded by the limit: a caller
    /// that has finished a slice moves the cursor past it before widening or
    /// dropping the bound.
    pub fn set_limit(&mut self, end: usize) {
        self.limit = end.min(self.bytes.len());
    }

    /// Current byte offset.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// True when all input has been consumed.
    pub fn is_done(&self) -> bool {
        self.pos >= self.limit
    }

    /// Look at the current byte as a `char` without consuming it.
    pub fn peek(&self) -> Option<char> {
        self.byte_at(self.pos).map(|b| b as char)
    }

    /// The byte at `at`, or `None` at or past the limit.
    fn byte_at(&self, at: usize) -> Option<u8> {
        if at < self.limit {
            self.bytes.get(at).copied()
        } else {
            None
        }
    }

    /// Look at the byte one position beyond the cursor as a `char`, without
    /// consuming anything.
    ///
    /// `None` when the cursor is already on the last byte or past the end.
    /// This is the one-character lookahead a two-token decision needs (is
    /// this `[` the start of a bracket atom or of a bonding descriptor?);
    /// [`Scanner::peek`] reads the byte *at* the cursor.
    pub fn peek_next(&self) -> Option<char> {
        self.byte_at(self.pos + 1).map(|b| b as char)
    }

    /// Consume the current byte and return it as a `char`.
    pub fn advance(&mut self) -> Option<char> {
        let ch = self.byte_at(self.pos)? as char;
        self.pos += 1;
        Some(ch)
    }

    /// Consume the current byte if it matches `expected`, otherwise return an error.
    ///
    /// # Errors
    ///
    /// Returns [`SmilesErrorKind::UnexpectedChar`] when another byte sits at
    /// the cursor and [`SmilesErrorKind::UnexpectedEnd`] at end of input; in
    /// both cases the cursor does not move.
    pub fn expect(&mut self, expected: char) -> Result<(), SmilesError> {
        match self.peek() {
            Some(c) if c == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(c) => Err(self.error(SmilesErrorKind::UnexpectedChar(c))),
            None => Err(self.error(SmilesErrorKind::UnexpectedEnd)),
        }
    }

    /// The whole input the scanner runs over.
    ///
    /// The scanner already borrows the string it scans, so a caller that needs
    /// token text (`&scanner.input()[start..scanner.pos()]`) or the error
    /// context a [`SmilesError`] carries takes it from here instead of holding
    /// a second copy of the same `&str` alongside the scanner. The returned
    /// borrow lives as long as the input, not as long as the scanner.
    pub fn input(&self) -> &'a str {
        self.input
    }

    /// Consume and return a run of ASCII digits. Returns an empty slice if
    /// the current byte is not a digit.
    pub fn eat_digits(&mut self) -> &'a str {
        let start = self.pos;
        while self.byte_at(self.pos).is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        &self.input[start..self.pos]
    }

    /// Consume a single ASCII digit and return its numeric value.
    pub fn eat_digit(&mut self) -> Option<u8> {
        match self.peek() {
            Some(c) if c.is_ascii_digit() => {
                self.pos += 1;
                Some(c as u8 - b'0')
            }
            _ => None,
        }
    }

    /// Move the cursor to byte offset `pos`, forwards or backwards.
    ///
    /// Re-reading a range of the input — what the `CGsmiles` `|n` repeat
    /// operator does with the unit it copies — rewinds this scanner instead of
    /// building a second one over a substring, so positions and error context
    /// stay relative to the whole input.
    ///
    /// # Invariant
    ///
    /// `pos` is **clamped** to the input length: `seek(len + k)` lands on
    /// `len`, the end-of-input offset. The clamp is unconditional, not a
    /// `debug_assert!`, so debug and release behave identically and the
    /// out-of-range case is testable by value. It is what keeps
    /// [`Scanner::eat_digits`]' `&self.input[start..self.pos]` in range, and
    /// what makes every [`Scanner::span_from`] minted after a rewind satisfy
    /// `start <= end` (the caller re-records the start after seeking).
    pub fn seek(&mut self, pos: usize) {
        self.pos = pos.min(self.bytes.len());
    }

    /// Build a [`Span`] from `start` to the current position.
    pub fn span_from(&self, start: usize) -> Span {
        Span::new(start, self.pos)
    }

    /// Build a [`SmilesError`] at the current position.
    ///
    /// The span is the single byte under the cursor, the input is the whole
    /// string this scanner runs over — so the rendered caret lands under the
    /// right column even after a rewind — and the notation is the one given to
    /// [`Scanner::new`], stamped here so no call site has to repeat it. At end
    /// of input the span still reads `pos..pos + 1`, one past the last byte;
    /// `SmilesError`'s `Display` clamps the caret to the input length.
    pub fn error(&self, kind: SmilesErrorKind) -> SmilesError {
        SmilesError::new(
            kind,
            Span::new(self.pos, self.pos + 1),
            self.input,
            self.notation,
        )
    }

    /// Build a [`SmilesError`] with a custom span.
    ///
    /// Same input and same notation stamp as [`Scanner::error`], for the
    /// errors that point at a range the cursor has already moved past — a
    /// whole bracket, a ring marker, the annotation field that was refused.
    pub fn error_at(&self, kind: SmilesErrorKind, span: Span) -> SmilesError {
        SmilesError::new(kind, span, self.input, self.notation)
    }

    /// Peek at the raw byte at the current position (for two-character symbol lookahead).
    pub fn peek_byte(&self) -> Option<u8> {
        self.byte_at(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::line_notation::error::Notation;

    #[test]
    fn test_peek_advance() {
        let mut s = Scanner::new("ABC", Notation::Smiles);
        assert_eq!(s.peek(), Some('A'));
        assert_eq!(s.advance(), Some('A'));
        assert_eq!(s.peek(), Some('B'));
        assert_eq!(s.advance(), Some('B'));
        assert_eq!(s.advance(), Some('C'));
        assert_eq!(s.advance(), None);
        assert!(s.is_done());
    }

    #[test]
    fn test_peek_next() {
        let mut s = Scanner::new("ab", Notation::Smiles);
        assert_eq!(s.peek_next(), Some('b'));
        s.advance();
        assert_eq!(s.peek_next(), None);
        s.advance();
        assert_eq!(s.peek_next(), None);
    }

    #[test]
    fn test_empty_input() {
        let s = Scanner::new("", Notation::Smiles);
        assert!(s.is_done());
        assert_eq!(s.peek(), None);
    }

    #[test]
    fn test_expect_ok() {
        let mut s = Scanner::new("C=O", Notation::Smiles);
        assert!(s.expect('C').is_ok());
        assert_eq!(s.pos(), 1);
    }

    #[test]
    fn test_expect_fail() {
        let mut s = Scanner::new("C=O", Notation::Smiles);
        let err = s.expect('N').unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnexpectedChar('C')));
    }

    #[test]
    fn test_expect_eof() {
        let mut s = Scanner::new("", Notation::Smiles);
        let err = s.expect('C').unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnexpectedEnd));
    }

    #[test]
    fn test_eat_digits() {
        let mut s = Scanner::new("123abc", Notation::Smiles);
        assert_eq!(s.eat_digits(), "123");
        assert_eq!(s.pos(), 3);
        assert_eq!(s.eat_digits(), "");
        assert_eq!(s.pos(), 3);
    }

    #[test]
    fn test_eat_digit() {
        let mut s = Scanner::new("5X", Notation::Smiles);
        assert_eq!(s.eat_digit(), Some(5));
        assert_eq!(s.eat_digit(), None);
        assert_eq!(s.pos(), 1);
    }

    #[test]
    fn test_span_from() {
        let mut s = Scanner::new("ABCDEF", Notation::Smiles);
        s.advance();
        s.advance();
        let span = s.span_from(0);
        assert_eq!(span, Span::new(0, 2));
    }

    #[test]
    fn test_pos_tracking() {
        let mut s = Scanner::new("C(=O)O", Notation::Smiles);
        for _ in 0..6 {
            s.advance();
        }
        assert_eq!(s.pos(), 6);
        assert!(s.is_done());
    }

    // -- seek ---------------------------------------------------------------

    #[test]
    fn test_seek_back_rereads_the_same_bytes() {
        let mut s = Scanner::new("ABCDEF", Notation::Smiles);
        s.advance();
        s.advance();
        s.advance();
        s.seek(1);
        assert_eq!(s.advance(), Some('B'));
    }

    #[test]
    fn test_error_after_seek_reports_the_position_in_the_full_input() {
        let mut s = Scanner::new("ABCDEF", Notation::Smiles);
        for _ in 0..6 {
            s.advance();
        }
        s.seek(4);
        let err = s.error(SmilesErrorKind::UnexpectedEnd);
        assert_eq!(err.span.start, 4);
        assert_eq!(err.input, "ABCDEF");
    }

    #[test]
    fn test_seek_past_the_end_lands_on_the_last_byte_offset() {
        let mut s = Scanner::new("ABC", Notation::Smiles);
        s.seek(4);
        assert_eq!(s.pos(), 3);
    }

    #[test]
    fn test_seek_past_the_end_keeps_eat_digits_in_range() {
        let mut s = Scanner::new("123", Notation::Smiles);
        s.seek(4);
        assert_eq!(s.eat_digits(), "");
    }

    #[test]
    fn test_span_minted_after_a_backward_seek_is_well_formed() {
        let mut s = Scanner::new("ABCDEF", Notation::Smiles);
        for _ in 0..5 {
            s.advance();
        }
        s.seek(2);
        let start = s.pos();
        s.advance();
        s.advance();
        let span = s.span_from(start);
        assert!(span.start <= span.end, "span was {span:?}");
    }

    #[test]
    fn test_error_carries_the_scanner_notation() {
        let s = Scanner::new("{[#A]}", Notation::CgSmiles);
        let err = s.error(SmilesErrorKind::UnexpectedEnd);
        assert_eq!(err.notation, Notation::CgSmiles);
    }
}
