//! Recursive-descent parser for compound unit expressions.
//!
//! Grammar:
//! ```text
//! expr     := term (('*' | '/' | '·' | ' ') term)*
//! term     := factor ('^' | '**')? signed_int?
//! factor   := IDENT | NUMBER | '(' expr ')'
//! ```

use crate::op::F;

use super::dimension::Dimension;
use super::error::UnitsError;
use super::registry::UnitRegistry;
use super::unit::Unit;

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Ident(String),
    Number(F),
    Mul,
    Div,
    Pow,
    Minus,
    LParen,
    RParen,
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '°' || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

fn parse_error(expr: &str, message: impl Into<String>) -> UnitsError {
    UnitsError::Parse {
        expr: expr.to_string(),
        message: message.into(),
    }
}

fn tokenize(expr: &str) -> Result<Vec<Token>, UnitsError> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        match c {
            '*' => {
                chars.next();
                if chars.peek() == Some(&'*') {
                    chars.next();
                    tokens.push(Token::Pow);
                } else {
                    tokens.push(Token::Mul);
                }
            }
            '·' => {
                chars.next();
                tokens.push(Token::Mul);
            }
            '/' => {
                chars.next();
                tokens.push(Token::Div);
            }
            '^' => {
                chars.next();
                tokens.push(Token::Pow);
            }
            '-' => {
                chars.next();
                tokens.push(Token::Minus);
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            c if is_ident_start(c) => {
                let mut ident = String::new();
                while let Some(&c) = chars.peek() {
                    if !is_ident_continue(c) {
                        break;
                    }
                    ident.push(c);
                    chars.next();
                }
                tokens.push(Token::Ident(ident));
            }
            c if c.is_ascii_digit() || c == '.' => {
                let mut num = String::new();
                while let Some(&c) = chars.peek() {
                    if !c.is_ascii_digit() && c != '.' {
                        break;
                    }
                    num.push(c);
                    chars.next();
                }
                let value: F = num
                    .parse()
                    .map_err(|_| parse_error(expr, format!("invalid number '{}'", num)))?;
                tokens.push(Token::Number(value));
            }
            other => {
                return Err(parse_error(
                    expr,
                    format!("unexpected character '{}'", other),
                ));
            }
        }
    }
    Ok(tokens)
}

/// Folded result of evaluating a (sub-)expression: a multiplicative factor,
/// a dimension, the named atoms with merged exponents, and the product of
/// bare numeric literals (`numeric` is part of `factor` but tracked
/// separately so the canonical display name can preserve it).
struct Acc {
    factor: F,
    dim: Dimension,
    atoms: Vec<(String, i32)>,
    numeric: F,
}

impl Acc {
    fn combine(mut self, rhs: Acc, divide: bool) -> Acc {
        if divide {
            self.factor /= rhs.factor;
            self.numeric /= rhs.numeric;
            self.dim = self.dim / rhs.dim;
            self.atoms
                .extend(rhs.atoms.into_iter().map(|(n, e)| (n, -e)));
        } else {
            self.factor *= rhs.factor;
            self.numeric *= rhs.numeric;
            self.dim = self.dim * rhs.dim;
            self.atoms.extend(rhs.atoms);
        }
        self
    }
}

/// Canonical display name: leading numeric literal (when ≠ 1), then atoms
/// merged by first occurrence with zero exponents dropped, rendered as
/// `name` / `name^exp` joined by ` * `. Idempotent under re-parsing (the
/// parse round-trip contract); `f64` `Display` never emits scientific
/// notation, so the numeric part stays tokenizable.
fn render(numeric: F, atoms: &[(String, i32)]) -> String {
    let mut merged: Vec<(String, i32)> = Vec::new();
    for (name, exp) in atoms {
        if let Some(entry) = merged.iter_mut().find(|(n, _)| n == name) {
            entry.1 += exp;
        } else {
            merged.push((name.clone(), *exp));
        }
    }
    merged.retain(|(_, e)| *e != 0);
    let mut parts: Vec<String> = Vec::with_capacity(merged.len() + 1);
    if numeric != 1.0 {
        parts.push(format!("{}", numeric));
    }
    parts.extend(merged.iter().map(|(name, exp)| {
        if *exp == 1 {
            name.clone()
        } else {
            format!("{}^{}", name, exp)
        }
    }));
    if parts.is_empty() {
        return "1".to_string();
    }
    parts.join(" * ")
}

struct Parser<'a> {
    registry: &'a UnitRegistry,
    expr: &'a str,
    tokens: Vec<Token>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    /// expr := term (('*' | '/' | implicit) term)*
    fn expr(&mut self) -> Result<Acc, UnitsError> {
        let mut acc = self.term()?;
        loop {
            match self.peek() {
                Some(Token::Mul) => {
                    self.next();
                    let rhs = self.term()?;
                    acc = acc.combine(rhs, false);
                }
                Some(Token::Div) => {
                    self.next();
                    let rhs = self.term()?;
                    acc = acc.combine(rhs, true);
                }
                // Implicit multiplication: "m s^-2", "kg (m/s)".
                Some(Token::Ident(_)) | Some(Token::Number(_)) | Some(Token::LParen) => {
                    let rhs = self.term()?;
                    acc = acc.combine(rhs, false);
                }
                _ => break,
            }
        }
        Ok(acc)
    }

    /// term := factor (('^' | '**') signed_int)?
    fn term(&mut self) -> Result<Acc, UnitsError> {
        let mut acc = self.factor()?;
        if self.peek() == Some(&Token::Pow) {
            self.next();
            let exp = self.signed_int()?;
            acc.factor = acc.factor.powi(exp);
            acc.numeric = acc.numeric.powi(exp);
            acc.dim = acc.dim.pow(exp);
            for (_, e) in &mut acc.atoms {
                *e *= exp;
            }
        }
        Ok(acc)
    }

    /// factor := IDENT | NUMBER | '(' expr ')'
    fn factor(&mut self) -> Result<Acc, UnitsError> {
        match self.next() {
            Some(Token::Ident(name)) => {
                let (factor, offset, dim) = self
                    .registry
                    .resolve_atom(&name)
                    .ok_or(UnitsError::UnknownUnit { name: name.clone() })?;
                if offset != 0.0 {
                    return Err(UnitsError::AffineUnit {
                        name,
                        operation: "compound expression",
                    });
                }
                Ok(Acc {
                    factor,
                    dim,
                    atoms: vec![(name, 1)],
                    numeric: 1.0,
                })
            }
            Some(Token::Number(value)) => Ok(Acc {
                factor: value,
                dim: Dimension::DIMENSIONLESS,
                atoms: vec![],
                numeric: value,
            }),
            Some(Token::LParen) => {
                let acc = self.expr()?;
                match self.next() {
                    Some(Token::RParen) => Ok(acc),
                    _ => Err(parse_error(self.expr, "missing closing parenthesis")),
                }
            }
            Some(tok) => Err(parse_error(
                self.expr,
                format!("unexpected token {:?}", tok),
            )),
            None => Err(parse_error(self.expr, "unexpected end of expression")),
        }
    }

    fn signed_int(&mut self) -> Result<i32, UnitsError> {
        let negative = if self.peek() == Some(&Token::Minus) {
            self.next();
            true
        } else {
            false
        };
        match self.next() {
            Some(Token::Number(value)) if value.fract() == 0.0 && value.abs() <= i32::MAX as F => {
                let n = value as i32;
                Ok(if negative { -n } else { n })
            }
            Some(tok) => Err(parse_error(
                self.expr,
                format!("expected integer exponent, found {:?}", tok),
            )),
            None => Err(parse_error(self.expr, "missing exponent")),
        }
    }
}

/// Parse `expr` against `registry`, returning a resolved [`Unit`].
pub(crate) fn parse_expr(registry: &UnitRegistry, expr: &str) -> Result<Unit, UnitsError> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Err(parse_error(expr, "empty unit expression"));
    }
    // Single bare atom: the only spelling in which an affine unit is legal.
    if trimmed.chars().enumerate().all(|(i, c)| {
        if i == 0 {
            is_ident_start(c)
        } else {
            is_ident_continue(c)
        }
    }) {
        let (factor, offset, dim) =
            registry
                .resolve_atom(trimmed)
                .ok_or_else(|| UnitsError::UnknownUnit {
                    name: trimmed.to_string(),
                })?;
        return Ok(Unit::new(factor, offset, dim, trimmed.to_string()));
    }
    let tokens = tokenize(trimmed)?;
    let mut parser = Parser {
        registry,
        expr: trimmed,
        tokens,
        pos: 0,
    };
    let acc = parser.expr()?;
    if parser.peek().is_some() {
        return Err(parse_error(trimmed, "trailing tokens after expression"));
    }
    Ok(Unit::new(
        acc.factor,
        0.0,
        acc.dim,
        render(acc.numeric, &acc.atoms),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Dimension;

    fn reg() -> UnitRegistry {
        UnitRegistry::new()
    }

    #[test]
    fn parses_bare_atom() {
        let u = reg().parse("meter").unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH);
    }

    #[test]
    fn parses_alias() {
        let u = reg().parse("Å").unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH);
    }

    #[test]
    fn parses_short_alias_ang() {
        let u = reg().parse("ang").unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH);
    }

    #[test]
    fn parses_prefix_femto_second() {
        let r = reg();
        let fs = r.parse("fs").unwrap();
        let s = r.parse("s").unwrap();
        // 1 fs = 1e-15 s
        let factor = fs.factor_to(&s).unwrap();
        assert!((factor - 1e-15).abs() < 1e-29, "fs->s factor = {}", factor);
    }

    #[test]
    fn parses_prefix_kcal() {
        let u = reg().parse("kcal").unwrap();
        assert_eq!(u.dimension(), Dimension::ENERGY);
    }

    #[test]
    fn parses_prefix_nm() {
        let u = reg().parse("nm").unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH);
    }

    #[test]
    fn parses_micro_prefix_us() {
        let u = reg().parse("µs").unwrap();
        assert_eq!(u.dimension(), Dimension::TIME);
    }

    #[test]
    fn parses_caret_exponent() {
        let u = reg().parse("m^2").unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH.pow(2));
    }

    #[test]
    fn parses_double_star_exponent() {
        let u = reg().parse("m**2").unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH.pow(2));
    }

    #[test]
    fn parses_negative_exponent() {
        let u = reg().parse("s^-2").unwrap();
        assert_eq!(u.dimension(), Dimension::TIME.pow(-2));
    }

    #[test]
    fn parses_parentheses() {
        let u = reg().parse("kg/(m s^2)").unwrap();
        assert_eq!(u.dimension(), Dimension::PRESSURE);
    }

    #[test]
    fn parses_implicit_multiply_with_space() {
        // "m s^-2" = acceleration: L T⁻²
        let u = reg().parse("m s^-2").unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH * Dimension::TIME.pow(-2));
    }

    #[test]
    fn parses_middot_separator() {
        let u = reg().parse("kg·m^2").unwrap();
        assert_eq!(u.dimension(), Dimension::MASS * Dimension::LENGTH.pow(2));
    }

    #[test]
    fn parses_compound_division_chain() {
        // kcal/mol/angstrom = ENERGY/AMOUNT/LENGTH
        let u = reg().parse("kcal/mol/angstrom").unwrap();
        let expected = Dimension::ENERGY / Dimension::AMOUNT / Dimension::LENGTH;
        assert_eq!(u.dimension(), expected);
    }

    #[test]
    fn error_unknown_unit() {
        let err = reg().parse("zorp").unwrap_err();
        assert!(matches!(err, UnitsError::UnknownUnit { .. }));
    }

    #[test]
    fn error_bad_exponent() {
        let err = reg().parse("m^").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }));
    }

    #[test]
    fn error_empty_string() {
        let err = reg().parse("").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }));
    }

    #[test]
    fn error_affine_in_compound() {
        // °C may not appear in a compound expression.
        let err = reg().parse("degC m").unwrap_err();
        assert!(matches!(err, UnitsError::AffineUnit { .. }));
    }

    #[test]
    fn error_double_prefix_kkg_rejected() {
        // "kkg" must not resolve via double prefix.
        let err = reg().parse("kkg").unwrap_err();
        assert!(matches!(err, UnitsError::UnknownUnit { .. }));
    }

    // -- malformed / extreme inputs (review phase 4) -----------------------

    #[test]
    fn error_leading_caret() {
        // Exponent operator with no base.
        let err = reg().parse("^2").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn error_empty_parentheses() {
        let err = reg().parse("()").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn error_exponent_exceeds_i32() {
        // 99999999999 > i32::MAX must be rejected, not wrapped/truncated.
        let err = reg().parse("m^99999999999").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn error_fractional_exponent() {
        let err = reg().parse("m^2.5").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn error_minus_without_exponent() {
        let err = reg().parse("m^-").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn error_missing_closing_paren() {
        let err = reg().parse("(m").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn error_whitespace_only() {
        // Whitespace-only input trims to the empty expression.
        let err = reg().parse("   ").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn error_unicode_alphabetic_unknown_unit() {
        // CJK char is alphabetic → lexes as an ident → unknown unit.
        let err = reg().parse("米").unwrap_err();
        assert!(matches!(err, UnitsError::UnknownUnit { .. }), "got {err:?}");
    }

    #[test]
    fn error_non_alphabetic_unicode_char() {
        // Emoji is neither ident nor operator → tokenizer Parse error.
        let err = reg().parse("🚀").unwrap_err();
        assert!(matches!(err, UnitsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn parses_nested_parentheses() {
        let u = reg().parse("((m))").unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH);
        assert_eq!(u.factor, 1.0);
    }

    #[test]
    fn parses_one_over_s() {
        // Numeric numerator: "1/s" = frequency, factor exactly 1.
        let u = reg().parse("1/s").unwrap();
        assert_eq!(u.dimension(), Dimension::TIME.pow(-1));
        assert_eq!(u.factor, 1.0);
    }

    #[test]
    fn parses_bare_number_as_dimensionless_factor() {
        // Grammar admits NUMBER as a factor: "2" is a dimensionless unit
        // with factor 2 (pint parity).
        let u = reg().parse("2").unwrap();
        assert!(u.dimension().is_dimensionless());
        assert_eq!(u.factor, 2.0);
    }

    #[test]
    fn canceling_atoms_render_as_one() {
        // m/m: exponents cancel, canonical display collapses to "1".
        let u = reg().parse("m/m").unwrap();
        assert!(u.dimension().is_dimensionless());
        assert_eq!(u.factor, 1.0);
        assert_eq!(u.to_string(), "1");
    }

    #[test]
    fn parses_degree_symbol_celsius_bare_atom() {
        // '°' is an ident-start char; bare "°C" is the only legal affine spelling.
        let u = reg().parse("°C").unwrap();
        assert!(u.is_affine());
        assert_eq!(u.dimension(), Dimension::TEMPERATURE);
    }
}
