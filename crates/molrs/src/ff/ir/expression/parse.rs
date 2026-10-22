//! A hand-written lexer and precedence-climbing parser for molrec's Lepton
//! subset.

use super::ast::{BinOp, Definition, Expr, NEG_PRECEDENCE, Parsed};
use super::error::ExpressionError;
use molrs::op::F;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(F),
    Ident(String),
    Op(BinOp),
    LParen,
    RParen,
    Comma,
    Semi,
    Eq,
}

impl Tok {
    fn text(&self) -> String {
        match self {
            Tok::Num(v) => v.to_string(),
            Tok::Ident(s) => s.clone(),
            Tok::Op(op) => op.symbol().to_string(),
            Tok::LParen => "(".into(),
            Tok::RParen => ")".into(),
            Tok::Comma => ",".into(),
            Tok::Semi => ";".into(),
            Tok::Eq => "=".into(),
        }
    }
}

fn lex(src: &str) -> Result<Vec<(usize, Tok)>, ExpressionError> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let start = i;
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => {
                i += 1;
                continue;
            }
            b'+' | b'-' | b'*' | b'/' | b'^' => {
                let op = match c {
                    b'+' => BinOp::Add,
                    b'-' => BinOp::Sub,
                    b'*' => BinOp::Mul,
                    b'/' => BinOp::Div,
                    _ => BinOp::Pow,
                };
                out.push((start, Tok::Op(op)));
                i += 1;
            }
            b'(' => {
                out.push((start, Tok::LParen));
                i += 1;
            }
            b')' => {
                out.push((start, Tok::RParen));
                i += 1;
            }
            b',' => {
                out.push((start, Tok::Comma));
                i += 1;
            }
            b';' => {
                out.push((start, Tok::Semi));
                i += 1;
            }
            b'=' => {
                out.push((start, Tok::Eq));
                i += 1;
            }
            b'0'..=b'9' | b'.' => {
                let mut digits = 0;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                    digits += 1;
                }
                if i < bytes.len() && bytes[i] == b'.' {
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                        digits += 1;
                    }
                }
                if digits == 0 {
                    return Err(ExpressionError::UnexpectedChar {
                        pos: start,
                        ch: '.',
                    });
                }
                // An exponent only when digits follow it; `2e` is a number
                // then an identifier (and so a syntax error later).
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    let mut j = i + 1;
                    if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                        j += 1;
                    }
                    if j < bytes.len() && bytes[j].is_ascii_digit() {
                        while j < bytes.len() && bytes[j].is_ascii_digit() {
                            j += 1;
                        }
                        i = j;
                    }
                }
                let text = &src[start..i];
                let v: F = text.parse().map_err(|_| ExpressionError::UnexpectedToken {
                    pos: start,
                    found: text.to_owned(),
                    expected: "a number",
                })?;
                out.push((start, Tok::Num(v)));
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                out.push((start, Tok::Ident(src[start..i].to_owned())));
            }
            _ => {
                let ch = src[start..].chars().next().unwrap_or('?');
                return Err(ExpressionError::UnexpectedChar { pos: start, ch });
            }
        }
    }
    Ok(out)
}

struct Parser<'a> {
    toks: &'a [(usize, Tok)],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|(_, t)| t)
    }

    fn unexpected(&self, expected: &'static str) -> ExpressionError {
        match self.toks.get(self.pos) {
            Some((p, t)) => ExpressionError::UnexpectedToken {
                pos: *p,
                found: t.text(),
                expected,
            },
            None => ExpressionError::UnexpectedEnd { expected },
        }
    }

    fn expr(&mut self, min_prec: u8) -> Result<Expr, ExpressionError> {
        let mut lhs = self.prefix()?;
        while let Some(Tok::Op(op)) = self.peek() {
            let op = *op;
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            self.pos += 1;
            let next = if op.right_associative() {
                prec
            } else {
                prec + 1
            };
            let rhs = self.expr(next)?;
            lhs = Expr::bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn prefix(&mut self) -> Result<Expr, ExpressionError> {
        const OPERAND: &str = "a number, a name, `(` or `-`";
        let Some((_, tok)) = self.toks.get(self.pos) else {
            return Err(ExpressionError::UnexpectedEnd { expected: OPERAND });
        };
        match tok {
            Tok::Op(BinOp::Sub) => {
                self.pos += 1;
                Ok(Expr::neg(self.expr(NEG_PRECEDENCE)?))
            }
            Tok::Num(v) => {
                self.pos += 1;
                Ok(Expr::Num(*v))
            }
            Tok::Ident(name) => {
                self.pos += 1;
                if self.peek() != Some(&Tok::LParen) {
                    return Ok(Expr::Var(name.clone()));
                }
                self.pos += 1;
                let mut args = Vec::new();
                if self.peek() == Some(&Tok::RParen) {
                    self.pos += 1;
                    return Ok(Expr::Call(name.clone(), args));
                }
                loop {
                    args.push(self.expr(0)?);
                    match self.peek() {
                        Some(Tok::Comma) => self.pos += 1,
                        Some(Tok::RParen) => {
                            self.pos += 1;
                            return Ok(Expr::Call(name.clone(), args));
                        }
                        _ => return Err(self.unexpected("`,` or `)`")),
                    }
                }
            }
            Tok::LParen => {
                self.pos += 1;
                let e = self.expr(0)?;
                if self.peek() != Some(&Tok::RParen) {
                    return Err(self.unexpected("`)`"));
                }
                self.pos += 1;
                Ok(e)
            }
            _ => Err(self.unexpected(OPERAND)),
        }
    }

    /// A whole segment as one expression: anything left over is an error.
    fn segment(&mut self) -> Result<Expr, ExpressionError> {
        let e = self.expr(0)?;
        if self.pos < self.toks.len() {
            return Err(self.unexpected("an operator or the end"));
        }
        Ok(e)
    }
}

/// Parse an expression. The result keeps `src` byte for byte.
///
/// The grammar is molrec's (`docs/spec/forcefield.md` § Expressions,
/// normative), which is Lepton's precedence and associativity:
///
/// ```text
/// expression := formula (";" definition)*
/// definition := name "=" formula
/// formula    := term (("+" | "-") term)*                  left-associative
/// term       := factor (("*" | "/") factor)*              left-associative
/// factor     := "-" factor | power
/// power      := primary ("^" factor)?                     right-associative
/// primary    := number | name | call | "(" formula ")"
/// call       := name "(" formula ("," formula)* ")"
/// number     := digits ["." [digits]] [exponent] | "." digits [exponent]
/// exponent   := ("e" | "E") ["+" | "-"] digits
/// name       := [A-Za-z_][A-Za-z0-9_]*
/// ```
///
/// So `-x^2` is `-(x^2)`, `-x*y` is `(-x)*y`, `2^3^2` is `2^(3^2)` and
/// `x^-2` is `x^(-2)`. Whitespace between tokens is ignored. An empty
/// segment after a `;` (a trailing `;`) is no definition and is refused. A
/// call with no argument parses and is refused by the compiler as a wrong
/// number of arguments, naming the function.
pub fn parse(src: &str) -> Result<Parsed, ExpressionError> {
    let toks = lex(src)?;
    let mut segments = toks.split(|(_, t)| *t == Tok::Semi);
    let main_toks = segments.next().unwrap_or(&[]);
    if main_toks.is_empty() {
        return Err(ExpressionError::EmptyExpression);
    }
    let main = Parser {
        toks: main_toks,
        pos: 0,
    }
    .segment()?;

    let mut defs = Vec::new();
    for seg in segments {
        if seg.is_empty() {
            return Err(ExpressionError::UnexpectedEnd {
                expected: "a definition `name=formula` after `;`",
            });
        }
        let pos = seg[0].0;
        let name = match (&seg[0].1, seg.get(1).map(|(_, t)| t)) {
            (Tok::Ident(name), Some(Tok::Eq)) => name.clone(),
            _ => {
                let end = seg
                    .get(1)
                    .map(|(p, t)| p + t.text().len())
                    .unwrap_or(pos + seg[0].1.text().len());
                return Err(ExpressionError::BadDefinition {
                    pos,
                    text: src[pos..end.min(src.len())].to_owned(),
                });
            }
        };
        let body = &seg[2..];
        if body.is_empty() {
            return Err(ExpressionError::UnexpectedEnd {
                expected: "the definition's expression",
            });
        }
        let expr = Parser { toks: body, pos: 0 }.segment()?;
        defs.push(Definition { name, expr });
    }

    // An `=` anywhere but right after a definition's name is a syntax error,
    // which `segment` reports (an `=` is no operand and no operator).
    Ok(Parsed {
        source: src.to_owned(),
        main,
        defs,
    })
}
