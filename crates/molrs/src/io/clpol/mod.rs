//! CL&Pol's `alpha.ff`, the Drude polarisation table of the CL&Pol
//! polarisable force field (paduagroup/clandpol).
//!
//! The door is a function of [`crate::io`],
//! [`read_clpol_alpha`](crate::io::read_clpol_alpha), over a
//! [`Source`](crate::io::Source). This module holds the format's row,
//! [`ClpolAlphaRow`].

pub(crate) mod codec;

pub use codec::ClpolAlphaRow;
