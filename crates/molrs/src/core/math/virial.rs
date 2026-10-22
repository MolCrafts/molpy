//! The virial tensor, and the one thing about it that is easy to get wrong.

use crate::op::F;

/// The virial tensor `W = Σ f ⊗ r`, symmetric, in kcal/mol.
///
/// Stored as the six independent components in LAMMPS's order — `xx`, `yy`,
/// `zz`, `xy`, `xz`, `yz` — because the tensor is symmetric and carrying nine
/// numbers invites two of them to drift apart.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Virial {
    /// `[xx, yy, zz, xy, xz, yz]` in kcal/mol.
    pub components: [F; 6],
}

impl Virial {
    /// Zero.
    pub const ZERO: Self = Self {
        components: [0.0; 6],
    };

    /// Accumulate one pair's contribution, `f ⊗ r`.
    #[inline]
    pub fn add_outer(&mut self, f: [F; 3], r: [F; 3]) {
        self.components[0] += f[0] * r[0];
        self.components[1] += f[1] * r[1];
        self.components[2] += f[2] * r[2];
        self.components[3] += f[0] * r[1];
        self.components[4] += f[0] * r[2];
        self.components[5] += f[1] * r[2];
    }

    /// `W_xx + W_yy + W_zz`.
    pub fn trace(&self) -> F {
        self.components[0] + self.components[1] + self.components[2]
    }

    /// The scalar pressure `P = (2·K + tr W) / (3V)`.
    ///
    /// `kinetic` is the kinetic energy (kcal/mol) and `volume` is the cell
    /// volume (Å³), so the result is kcal·mol⁻¹·Å⁻³. Multiply by
    /// `69476.95` to read it in bar.
    pub fn pressure(&self, kinetic: F, volume: F) -> F {
        (2.0 * kinetic + self.trace()) / (3.0 * volume)
    }

    /// The full symmetric matrix, for callers that want all nine entries.
    pub fn matrix(&self) -> [[F; 3]; 3] {
        let [xx, yy, zz, xy, xz, yz] = self.components;
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    }
}

impl std::ops::AddAssign for Virial {
    fn add_assign(&mut self, rhs: Self) {
        for (a, b) in self.components.iter_mut().zip(rhs.components) {
            *a += b;
        }
    }
}
