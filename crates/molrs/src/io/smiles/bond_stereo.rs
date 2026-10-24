//! Cis/trans `/` and `\` as one direction on the bond.
//!
//! OpenSMILES reads the character in the direction the string traverses the
//! bond. The graph stores that character along the bond's endpoint order:
//! `"up"` is `/` from `nodes[0]` toward `nodes[1]`, and `"down"` is `\`. A
//! writer flips the glyph when it traverses the bond the other way. The two
//! digits of a ring bond are that one direction: `/` written toward the
//! partner is `\` written back, and the same character on both digits
//! disagrees.
//!
//! A simple alkene does not keep this label. Its double bond stores `E` / `Z`,
//! and the slashes are derived when the string is written. `"up"` / `"down"`
//! remains only where the slash is not that geometry: a ring digit that is
//! not an alkene substituent, or a cumulene end.

use crate::line_notation::ast::BondKind;
use molrs::core::NodeId;

/// `/` or `\`, the two characters that name a direction.
pub(crate) fn is_directional(kind: BondKind) -> bool {
    matches!(kind, BondKind::Up | BondKind::Down)
}

/// The opposite character. A non-directional kind is unchanged.
pub(crate) fn flip(kind: BondKind) -> BondKind {
    match kind {
        BondKind::Up => BondKind::Down,
        BondKind::Down => BondKind::Up,
        other => other,
    }
}

/// The character to write while traversing `from` → `to`.
///
/// `stored` is `"up"` or `"down"`, the character from `endpoints[0]` toward
/// `endpoints[1]`. Any other label is not this direction.
pub(crate) fn kind_along(
    stored: &str,
    from: NodeId,
    to: NodeId,
    endpoints: [NodeId; 2],
) -> Option<BondKind> {
    let forward = match stored {
        "up" => BondKind::Up,
        "down" => BondKind::Down,
        _ => return None,
    };
    if from == endpoints[0] && to == endpoints[1] {
        Some(forward)
    } else if from == endpoints[1] && to == endpoints[0] {
        Some(flip(forward))
    } else {
        None
    }
}

/// `"up"` / `"down"` for a character written while traversing `from` → `to`.
///
/// The label is the character in the `endpoints[0]` → `endpoints[1]` direction.
/// `add_bond(a, b)` stores `[a, b]`, so a character written in that direction
/// is stored unchanged.
pub(crate) fn label_along(
    kind: BondKind,
    from: NodeId,
    to: NodeId,
    endpoints: [NodeId; 2],
) -> Option<&'static str> {
    let traversing_up = match kind {
        BondKind::Up => true,
        BondKind::Down => false,
        _ => return None,
    };
    let up = if from == endpoints[0] && to == endpoints[1] {
        traversing_up
    } else if from == endpoints[1] && to == endpoints[0] {
        !traversing_up
    } else {
        return None;
    };
    Some(if up { "up" } else { "down" })
}
