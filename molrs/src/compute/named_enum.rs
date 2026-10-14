//! `named_enum!` — the one way a user choice that has a name becomes an enum.
//!
//! An enum declared through it carries its canonical names: `as_str`, the
//! `ALL` / `NAMES` tables, `Display`, and a `FromStr` that accepts exactly the
//! canonical names — no aliases, no case folding. Bindings parse user strings
//! through this `FromStr` and never `match` on a string themselves, so every
//! binding accepts the same spellings and reports the same error.

/// Declare an enum whose variants have canonical names.
///
/// ```ignore
/// named_enum! {
///     /// How a pair distance is measured.
///     pub enum HBondDistanceKind {
///         /// Donor heavy atom to acceptor.
///         DonorAcceptor => "donor_acceptor",
///         /// Hydrogen to acceptor.
///         HydrogenAcceptor => "hydrogen_acceptor",
///     }
/// }
/// ```
macro_rules! named_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $text:literal
            ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// Every canonical name, in declaration order.
            pub const NAMES: &'static [&'static str] = &[$($text),+];

            /// The canonical name of this variant.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)+
                }
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = $crate::compute::ComputeError;

            fn from_str(name: &str) -> Result<Self, Self::Err> {
                match name {
                    $($text => Ok(Self::$variant),)+
                    _ => Err($crate::compute::ComputeError::UnknownName {
                        kind: stringify!($name),
                        got: name.to_string(),
                        expected: Self::NAMES,
                    }),
                }
            }
        }
    };
}

pub(crate) use named_enum;

#[cfg(test)]
mod tests {
    use crate::compute::ComputeError;

    named_enum! {
        /// A test choice.
        #[derive(Default)]
        enum Flavor {
            /// The default.
            #[default]
            Plain => "plain",
            Spiced => "spiced",
        }
    }

    #[test]
    fn every_name_round_trips() {
        for &flavor in Flavor::ALL {
            assert_eq!(flavor.as_str().parse::<Flavor>().unwrap(), flavor);
            assert_eq!(flavor.to_string(), flavor.as_str());
        }
        assert_eq!(Flavor::NAMES, &["plain", "spiced"]);
        assert_eq!(Flavor::default(), Flavor::Plain);
    }

    /// Only canonical spellings parse; the error names the kind, what was
    /// given and every accepted name.
    #[test]
    fn unknown_names_are_rejected_with_the_accepted_list() {
        for bad in ["Plain", "PLAIN", "p", " plain", ""] {
            let err = bad.parse::<Flavor>().unwrap_err();
            let ComputeError::UnknownName {
                kind,
                got,
                expected,
            } = &err
            else {
                panic!("expected UnknownName, got {err:?}");
            };
            assert_eq!(*kind, "Flavor");
            assert_eq!(got, bad);
            assert_eq!(*expected, Flavor::NAMES);
            assert!(err.to_string().contains("plain, spiced"));
        }
    }
}
