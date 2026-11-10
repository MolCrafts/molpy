//! The outer grammar of `BigSMILES`: molecules, stochastic objects,
//! terminals, size distributions and system amounts. Unit bodies go to the
//! shared line-notation parser in its `BigSMILES` dialect.

use super::{BigSmilesIr, FragmentSpec, MoleculeSpec, ObjectSpec, PartSpec, TerminalSpec};
use crate::builder::{Amount, Distribution};
use crate::io::smiles::{
    BondKind, DescriptorKind, FragmentIr, Notation, SmilesError, SmilesErrorKind, Span,
};
use crate::line_notation::parser::{parse_bigsmiles_body, weight_list};

pub(super) fn parse(text: &str) -> Result<BigSmilesIr, SmilesError> {
    let reader = Reader { text };
    if text.is_empty() {
        return Err(reader.error(SmilesErrorKind::EmptyInput, Span::new(0, 0)));
    }
    let mut molecules = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let (molecule, next) = reader.molecule(pos)?;
        molecules.push(molecule);
        pos = next;
    }
    Ok(BigSmilesIr {
        molecules,
        span: Span::new(0, text.len()),
        input: text.to_owned(),
    })
}

struct Reader<'a> {
    text: &'a str,
}

/// A piece of a molecule before its SMILES is parsed: the SMILES bodies
/// need the terminals of the objects around them.
enum Raw {
    Smiles(usize, usize),
    Object(ObjectSpec),
}

impl Reader<'_> {
    fn error(&self, kind: SmilesErrorKind, span: Span) -> SmilesError {
        SmilesError::new(kind, span, self.text, Notation::BigSmiles)
    }

    fn byte(&self, at: usize) -> Option<u8> {
        self.text.as_bytes().get(at).copied()
    }

    fn skip_ws(&self, mut at: usize, end: usize) -> usize {
        while at < end && self.text.as_bytes()[at].is_ascii_whitespace() {
            at += 1;
        }
        at
    }

    fn trim(&self, mut start: usize, mut end: usize) -> (usize, usize) {
        start = self.skip_ws(start, end);
        while end > start && self.text.as_bytes()[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
        (start, end)
    }

    // -- molecule -------------------------------------------------------------

    /// One molecule from `start`, and the position after it and its `.`
    /// (and amount).
    fn molecule(&self, start: usize) -> Result<(MoleculeSpec, usize), SmilesError> {
        let mut raw = Vec::new();
        let mut pos = start;
        while let Some(b) = self.byte(pos) {
            match b {
                b'.' => break,
                b'{' => {
                    let (object, next) = self.object(pos)?;
                    raw.push(Raw::Object(object));
                    pos = next;
                }
                _ => {
                    let end = self.smiles_end(pos)?;
                    raw.push(Raw::Smiles(pos, end));
                    pos = end;
                }
            }
        }
        if raw.is_empty() {
            let kind = match self.byte(pos) {
                Some(b) => SmilesErrorKind::UnexpectedChar(char::from(b)),
                None => SmilesErrorKind::UnexpectedEnd,
            };
            return Err(self.error(kind, Span::new(pos, pos + 1)));
        }
        let span = Span::new(start, pos);
        let mut amount = None;
        if self.byte(pos) == Some(b'.') {
            pos += 1;
            if self.byte(pos) == Some(b'|') {
                let (a, next) = self.amount(pos)?;
                amount = Some(a);
                pos = next;
            }
        }
        let parts = self.parts(raw)?;
        Ok((
            MoleculeSpec {
                parts,
                amount,
                span,
            },
            pos,
        ))
    }

    /// Where the SMILES starting at `start` ends: at a stochastic object, at
    /// the `.` ending the molecule, or at the end of the string.
    fn smiles_end(&self, start: usize) -> Result<usize, SmilesError> {
        let bytes = self.text.as_bytes();
        let mut depth = 0usize;
        let mut pos = start;
        while pos < bytes.len() {
            match bytes[pos] {
                b'[' => {
                    if self.byte(pos + 1) == Some(b'#') {
                        return Err(self.error(
                            SmilesErrorKind::BigUnsupported(
                                "fragment declarations [#name]".to_owned(),
                            ),
                            Span::new(pos, pos + 2),
                        ));
                    }
                    // A bracket holds no structure; skip to its end.
                    match self.text[pos..].find(']') {
                        Some(off) => pos += off,
                        None => return Ok(bytes.len()),
                    }
                }
                b'(' => depth += 1,
                b')' => depth = depth.saturating_sub(1),
                b'{' if depth > 0 => {
                    return Err(self.error(
                        SmilesErrorKind::BigUnsupported(
                            "stochastic objects inside a branch".to_owned(),
                        ),
                        Span::new(pos, pos + 1),
                    ));
                }
                b'{' | b'.' => return Ok(pos),
                _ => {}
            }
            pos += 1;
        }
        Ok(pos)
    }

    /// Turn the raw pieces into parts: each SMILES body takes the terminals
    /// of the objects beside it as descriptors, and each terminal the bond
    /// symbol written between it and that SMILES.
    fn parts(&self, raw: Vec<Raw>) -> Result<Vec<PartSpec>, SmilesError> {
        let bond_at = |at: usize| self.byte(at).and_then(|b| bond_kind(char::from(b)));
        // Terminal texts and spans of the neighbours of each SMILES piece.
        let mut objects: Vec<Option<ObjectSpec>> = raw
            .iter()
            .map(|r| match r {
                Raw::Object(o) => Some(o.clone()),
                Raw::Smiles(..) => None,
            })
            .collect();
        for (i, r) in raw.iter().enumerate() {
            let Raw::Smiles(start, end) = *r else {
                continue;
            };
            if i > 0
                && let Some(Some(prev)) = objects.get_mut(i - 1)
                && let Some(t) = prev.right.as_mut()
            {
                t.order = bond_at(start);
            }
            if let Some(Some(next)) = objects.get_mut(i + 1)
                && let Some(t) = next.left.as_mut()
            {
                t.order = end.checked_sub(1).and_then(bond_at);
            }
        }
        let terminal_text =
            |object: &Option<ObjectSpec>, right: bool| -> Result<String, SmilesError> {
                let Some(object) = object else {
                    return Ok(String::new());
                };
                let terminal = if right { &object.right } else { &object.left };
                match terminal {
                    Some(t) => Ok(format!("[{}{}]", t.kind.as_str(), t.label)),
                    None => Err(self.error(
                        SmilesErrorKind::BigMalformedObject(
                            "an object joined to SMILES needs a non-empty terminal on that side"
                                .to_owned(),
                        ),
                        object.span,
                    )),
                }
            };
        // Every SMILES body first, while each neighbour is still in place.
        let mut bodies: Vec<Option<FragmentSpec>> = Vec::with_capacity(raw.len());
        for (i, r) in raw.iter().enumerate() {
            let Raw::Smiles(start, end) = *r else {
                bodies.push(None);
                continue;
            };
            let before = match i.checked_sub(1) {
                Some(p) => terminal_text(&objects[p], true)?,
                None => String::new(),
            };
            let after = match objects.get(i + 1) {
                Some(next) => terminal_text(next, false)?,
                None => String::new(),
            };
            bodies.push(Some(self.body(start, end, &before, &after)?));
        }
        Ok(objects
            .into_iter()
            .zip(bodies)
            .map(|pair| match pair {
                (Some(object), _) => PartSpec::Stochastic(object),
                (None, Some(body)) => PartSpec::Smiles(body),
                (None, None) => unreachable!("each piece is an object or SMILES"),
            })
            .collect())
    }

    // -- stochastic object ----------------------------------------------------

    /// The object whose `{` is at `open`, and the position after it (and
    /// its distribution).
    fn object(&self, open: usize) -> Result<(ObjectSpec, usize), SmilesError> {
        if self.byte(open + 1) == Some(b'#') {
            return Err(self.error(
                SmilesErrorKind::BigUnsupported("fragment definitions {#name=…}".to_owned()),
                Span::new(open, open + 2),
            ));
        }
        let close = self.object_close(open)?;
        let malformed = |reason: &str| {
            self.error(
                SmilesErrorKind::BigMalformedObject(reason.to_owned()),
                Span::new(open, close + 1),
            )
        };

        // The left terminal: the first bracket.
        let left_start = self.skip_ws(open + 1, close);
        if self.byte(left_start) != Some(b'[') {
            return Err(malformed("an object opens with a terminal descriptor"));
        }
        let left_end = left_start
            + self.text[left_start..close]
                .find(']')
                .ok_or_else(|| malformed("the left terminal is never closed"))?
            + 1;
        // The right terminal: the last bracket.
        let (_, inner_end) = self.trim(left_end, close);
        if inner_end == left_end || self.byte(inner_end - 1) != Some(b']') {
            return Err(malformed("an object closes with a terminal descriptor"));
        }
        let right_start = self.text[left_end..inner_end]
            .rfind('[')
            .map(|off| left_end + off)
            .ok_or_else(|| malformed("an object closes with a terminal descriptor"))?;
        let left = self.terminal(left_start, left_end)?;
        let right = self.terminal(right_start, inner_end)?;

        // Units between the terminals.
        let (repeat, ends) = self.split_units(left_end, right_start, Span::new(open, close + 1))?;
        if repeat.is_empty() {
            return Err(malformed("an object needs at least one repeat unit"));
        }
        let repeat_units = repeat
            .into_iter()
            .map(|(s, e)| self.body(s, e, "", ""))
            .collect::<Result<Vec<_>, _>>()?;
        let end_groups = ends
            .into_iter()
            .map(|(s, e)| self.body(s, e, "", ""))
            .collect::<Result<Vec<_>, _>>()?;

        let mut pos = close + 1;
        let mut distribution = None;
        if self.byte(pos) == Some(b'|') {
            let (d, next) = self.distribution(pos)?;
            distribution = Some(d);
            pos = next;
        }
        Ok((
            ObjectSpec {
                left,
                repeat_units,
                end_groups,
                right,
                distribution,
                span: Span::new(open, pos),
            },
            pos,
        ))
    }

    /// The `}` closing the object opened at `open`.
    fn object_close(&self, open: usize) -> Result<usize, SmilesError> {
        let bytes = self.text.as_bytes();
        let mut in_bracket = false;
        for (pos, &b) in bytes.iter().enumerate().skip(open + 1) {
            match b {
                b'[' => in_bracket = true,
                b']' => in_bracket = false,
                b'{' if !in_bracket => {
                    return Err(self.error(
                        SmilesErrorKind::BigUnsupported("nested stochastic objects".to_owned()),
                        Span::new(pos, pos + 1),
                    ));
                }
                b'}' if !in_bracket => return Ok(pos),
                _ => {}
            }
        }
        Err(self.error(
            SmilesErrorKind::BigMalformedObject("'{' is never closed".to_owned()),
            Span::new(open, bytes.len()),
        ))
    }

    /// The units between the terminals, split at `,` and the one `;` that
    /// starts the end groups; each trimmed of whitespace.
    #[allow(clippy::type_complexity)]
    fn split_units(
        &self,
        start: usize,
        end: usize,
        object: Span,
    ) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>), SmilesError> {
        let bytes = self.text.as_bytes();
        let mut repeat = Vec::new();
        let mut ends = Vec::new();
        let mut in_ends = false;
        let mut in_bracket = false;
        let mut item = start;
        let mut push = |s: usize, e: usize, in_ends: bool| -> Result<(), SmilesError> {
            let (s, e) = self.trim(s, e);
            if s == e {
                return Err(self.error(
                    SmilesErrorKind::BigMalformedObject("an empty unit".to_owned()),
                    object,
                ));
            }
            if in_ends {
                ends.push((s, e))
            } else {
                repeat.push((s, e))
            }
            Ok(())
        };
        for (pos, &b) in bytes.iter().enumerate().take(end).skip(start) {
            match b {
                b'[' => in_bracket = true,
                b']' => in_bracket = false,
                b',' if !in_bracket => {
                    push(item, pos, in_ends)?;
                    item = pos + 1;
                }
                b';' if !in_bracket => {
                    if in_ends {
                        return Err(self.error(
                            SmilesErrorKind::BigMalformedObject(
                                "more than one ';' in an object".to_owned(),
                            ),
                            Span::new(pos, pos + 1),
                        ));
                    }
                    push(item, pos, in_ends)?;
                    in_ends = true;
                    item = pos + 1;
                }
                _ => {}
            }
        }
        push(item, end, in_ends)?;
        Ok((repeat, ends))
    }

    /// A terminal bracket `[ … ]` spanning `start..end`; `None` for `[]`.
    fn terminal(&self, start: usize, end: usize) -> Result<Option<TerminalSpec>, SmilesError> {
        let inner = self.text[start + 1..end - 1].trim();
        if inner.is_empty() {
            return Ok(None);
        }
        let malformed = || {
            self.error(
                SmilesErrorKind::BigMalformedObject(
                    "a terminal is [], or [$], [<], [>] with an optional id and weights".to_owned(),
                ),
                Span::new(start, end),
            )
        };
        let kind = match inner.as_bytes()[0] {
            b'$' => DescriptorKind::Symmetric,
            b'<' => DescriptorKind::Left,
            b'>' => DescriptorKind::Right,
            _ => return Err(malformed()),
        };
        let rest = &inner[1..];
        let label_len = rest.bytes().take_while(u8::is_ascii_digit).count();
        let label = rest[..label_len].to_owned();
        let rest = &rest[label_len..];
        let weights = if rest.is_empty() {
            None
        } else if rest.len() >= 2 && rest.starts_with('|') && rest.ends_with('|') {
            let list = &rest[1..rest.len() - 1];
            Some(weight_list(list).ok_or_else(|| {
                self.error(
                    SmilesErrorKind::BigInvalidWeights(list.to_owned()),
                    Span::new(start, end),
                )
            })?)
        } else {
            return Err(malformed());
        };
        Ok(Some(TerminalSpec {
            kind,
            label,
            order: None,
            weights,
        }))
    }

    /// The size distribution `|name(args)|` at `start`, and the position
    /// after it.
    fn distribution(&self, start: usize) -> Result<(Distribution, usize), SmilesError> {
        let (inner, span, next) = self.bars(start)?;
        let text = inner.trim();
        let bad = |kind| self.error(kind, span);
        let (name, args) = text
            .strip_suffix(')')
            .and_then(|t| t.split_once('('))
            .ok_or_else(|| bad(SmilesErrorKind::BigUnknownDistribution(text.to_owned())))?;
        let name = name.trim();
        let mut values = Vec::new();
        let pieces: Vec<&str> = args.split(',').map(str::trim).collect();
        for (i, piece) in pieces.iter().enumerate() {
            // The grammar allows a trailing comma: `flory_schulz(0.1,)`.
            if piece.is_empty() && i + 1 == pieces.len() && i > 0 {
                continue;
            }
            values.push(
                piece
                    .parse::<f64>()
                    .map_err(|_| bad(SmilesErrorKind::BigInvalidNumber((*piece).to_owned())))?,
            );
        }
        let arity = |expected: usize| {
            if values.len() == expected {
                Ok(())
            } else {
                Err(bad(SmilesErrorKind::BigDistributionArity {
                    name: name.to_owned(),
                    expected,
                    found: values.len(),
                }))
            }
        };
        let distribution = match name {
            "flory_schulz" => {
                arity(1)?;
                Distribution::FlorySchulz { a: values[0] }
            }
            "schulz_zimm" => {
                arity(2)?;
                Distribution::SchulzZimm {
                    mw: values[0],
                    mn: values[1],
                }
            }
            "gauss" => {
                arity(2)?;
                Distribution::Gauss {
                    mu: values[0],
                    sigma: values[1],
                }
            }
            "uniform" => {
                arity(2)?;
                Distribution::Uniform {
                    low: values[0],
                    high: values[1],
                }
            }
            "log_normal" => {
                arity(2)?;
                Distribution::LogNormal {
                    mn: values[0],
                    d: values[1],
                }
            }
            "poisson" => {
                arity(1)?;
                Distribution::Poisson { n: values[0] }
            }
            other => {
                return Err(bad(SmilesErrorKind::BigUnknownDistribution(
                    other.to_owned(),
                )));
            }
        };
        distribution
            .validate()
            .map_err(|e| bad(SmilesErrorKind::BigInvalidNumber(e.to_string())))?;
        Ok((distribution, next))
    }

    /// The system amount `|N|` or `|p%|` at `start`, and the position after
    /// it.
    fn amount(&self, start: usize) -> Result<(Amount, usize), SmilesError> {
        let (inner, span, next) = self.bars(start)?;
        let text = inner.trim();
        let invalid = || self.error(SmilesErrorKind::BigInvalidAmount(text.to_owned()), span);
        let amount = match text.strip_suffix('%') {
            Some(p) => {
                let p: f64 = p.trim().parse().map_err(|_| invalid())?;
                if !(p > 0.0 && p < 100.0) {
                    return Err(invalid());
                }
                Amount::Percent(p)
            }
            None => {
                let t: f64 = text.parse().map_err(|_| invalid())?;
                if !(t.is_finite() && t > 0.0) {
                    return Err(invalid());
                }
                Amount::Total(t)
            }
        };
        Ok((amount, next))
    }

    /// The text between the bars `|…|` opening at `start`, their span, and
    /// the position after the closing bar.
    fn bars(&self, start: usize) -> Result<(&str, Span, usize), SmilesError> {
        let close = self.text[start + 1..]
            .find('|')
            .map(|off| start + 1 + off)
            .ok_or_else(|| {
                self.error(
                    SmilesErrorKind::UnexpectedEnd,
                    Span::new(start, self.text.len()),
                )
            })?;
        Ok((
            &self.text[start + 1..close],
            Span::new(start, close + 1),
            close + 1,
        ))
    }

    // -- unit bodies ----------------------------------------------------------

    /// The body `text[start..end]`, with `before` and `after` (terminal
    /// descriptors of neighbouring objects) written around it, parsed in the
    /// `BigSMILES` dialect.
    fn body(
        &self,
        start: usize,
        end: usize,
        before: &str,
        after: &str,
    ) -> Result<FragmentSpec, SmilesError> {
        let written = format!("{before}{}{after}", &self.text[start..end]);
        let rebase = |e: SmilesError| self.rebase(e, start, end, before.len());
        let (ir, weights) = parse_bigsmiles_body(&written).map_err(rebase)?;
        let ir = FragmentIr::new(ir);
        let (_, descriptors) = ir.to_atomistic().map_err(rebase)?;
        let weights = descriptors
            .iter()
            .map(|(_, d)| {
                weights
                    .iter()
                    .find(|(span, _)| *span == d.span)
                    .map(|(_, w)| w.clone())
            })
            .collect();
        Ok(FragmentSpec {
            ir,
            weights,
            span: Span::new(start, end),
        })
    }

    /// Re-base a body's diagnostic into the whole string: positions inside
    /// the written-around terminals map to the body's start, the rest shift
    /// by its offset.
    fn rebase(&self, err: SmilesError, start: usize, end: usize, before: usize) -> SmilesError {
        let at = |p: usize| (start + p.saturating_sub(before)).min(end);
        let span = Span::new(at(err.span.start), at(err.span.end));
        self.error(err.kind, span)
    }
}

/// The bond a SMILES bond symbol writes.
fn bond_kind(c: char) -> Option<BondKind> {
    Some(match c {
        '-' => BondKind::Single,
        '=' => BondKind::Double,
        '#' => BondKind::Triple,
        '$' => BondKind::Quadruple,
        ':' => BondKind::Aromatic,
        '/' => BondKind::Up,
        '\\' => BondKind::Down,
        _ => return None,
    })
}
