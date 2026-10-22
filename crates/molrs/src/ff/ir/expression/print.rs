//! The printer: a tree back to Lepton syntax, with the fewest parentheses
//! that parse back to the same tree.
//!
//! It exists for engine rewriting only (an OpenMM writer turning `r` into
//! `10*r`); a reader and a writer that do not rewrite carry
//! [`Parsed::source`](super::Parsed::source) byte for byte. Numbers print in
//! the shortest form that parses back to the same `f64`.

use std::fmt::{self, Write as _};

use super::ast::{Expr, NEG_PRECEDENCE, Parsed};
use molrs::op::F;

/// Precedence of an atom (number, name, call, parenthesised group).
const ATOM: u8 = 4;

fn precedence(e: &Expr) -> u8 {
    match e {
        Expr::Num(v) if v.is_sign_negative() && *v != 0.0 => NEG_PRECEDENCE,
        Expr::Num(_) | Expr::Var(_) | Expr::Call(..) => ATOM,
        Expr::Neg(_) => NEG_PRECEDENCE,
        Expr::Bin(op, ..) => op.precedence(),
    }
}

fn print_num(out: &mut String, v: F) {
    if v.is_nan() {
        out.push_str("(0/0)");
    } else if v.is_infinite() {
        out.push_str(if v > 0.0 { "(1/0)" } else { "(-1/0)" });
    } else if v == 0.0 || (1e-5..1e16).contains(&v.abs()) {
        // `{}` is the shortest digits that round-trip, without an exponent.
        let _ = write!(out, "{v}");
    } else {
        let _ = write!(out, "{v:e}");
    }
}

fn print_expr(out: &mut String, e: &Expr, min: u8) {
    let paren = precedence(e) < min;
    if paren {
        out.push('(');
    }
    match e {
        Expr::Num(v) => print_num(out, *v),
        Expr::Var(name) => out.push_str(name),
        Expr::Neg(inner) => {
            out.push('-');
            print_expr(out, inner, NEG_PRECEDENCE);
        }
        Expr::Bin(op, l, r) => {
            let p = op.precedence();
            // Left-associative: the left operand may sit at the same level;
            // right-associative `^`: the right one may.
            let (lmin, rmin) = if op.right_associative() {
                (p + 1, p)
            } else {
                (p, p + 1)
            };
            print_expr(out, l, lmin);
            out.push(op.symbol());
            print_expr(out, r, rmin);
        }
        Expr::Call(name, args) => {
            out.push_str(name);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                print_expr(out, a, 0);
            }
            out.push(')');
        }
    }
    if paren {
        out.push(')');
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        print_expr(&mut s, self, 0);
        f.write_str(&s)
    }
}

impl fmt::Display for Parsed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.main)?;
        for d in &self.defs {
            write!(f, "; {}={}", d.name, d.expr)?;
        }
        Ok(())
    }
}
