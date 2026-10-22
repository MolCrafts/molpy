//! A `.smi` file — one SMILES per line — as a [`FrameReader`].

use std::io::{BufRead, Result};

use crate::core::Atomistic;
use crate::core::Frame;
use crate::io::reader::{FrameReader, FromFrame, Reader};
use crate::io::smiles::SmilesIr;

/// Parse one SMILES string into an [`Atomistic`].
///
/// Stateless: the reader below is a thin cursor over lines, and this is where
/// the parsing lives, so `read` and `read_as` share one implementation rather
/// than each having their own.
fn parse_atomistic(smiles: &str) -> Result<Atomistic> {
    let ir = SmilesIr::parse(smiles)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    ir.to_atomistic()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

/// Reads one molecule per non-empty line.
///
/// SMILES parses to an [`Atomistic`] — that is what the notation *is*, a
/// connectivity graph — so the stateless parser returns one and
/// [`SmilesReader::read_as::<Atomistic>`] hands it straight over. Only
/// [`read`](FrameReader::read) pays for the `Atomistic -> Frame` conversion,
/// and only when the caller asked for a frame.
pub struct SmilesReader<R: BufRead> {
    reader: R,
}

impl<R: BufRead> Reader for SmilesReader<R> {
    type R = R;
    fn new(reader: Self::R) -> Self {
        SmilesReader { reader }
    }
}

impl<R: BufRead> SmilesReader<R> {
    /// Next non-empty, non-comment line, or `None` at EOF.
    fn next_record(&mut self) -> Result<Option<String>> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                return Ok(None);
            }
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                // A SMILES line may carry a trailing name field; the structure
                // is the first whitespace-delimited token.
                return Ok(Some(
                    trimmed.split_whitespace().next().unwrap_or("").to_string(),
                ));
            }
        }
    }
}

impl<R: BufRead> FrameReader for SmilesReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        match self.next_record()? {
            Some(s) => {
                let frame = parse_atomistic(&s)?.to_frame().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                })?;
                crate::io::reader::check_read_frame(Some(frame))
            }
            None => Ok(None),
        }
    }

    /// Overridden so asking for an [`Atomistic`] does not round-trip through a
    /// frame and back — the parser already produced exactly that.
    fn read_as<T: FromFrame>(&mut self) -> Result<Option<T>> {
        match self.next_record()? {
            Some(s) => {
                let mol = parse_atomistic(&s)?;
                Ok(Some(T::from_atomistic(mol)?))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::reader::FrameReader;

    fn reader(src: &str) -> SmilesReader<std::io::Cursor<Vec<u8>>> {
        SmilesReader::new(std::io::Cursor::new(src.as_bytes().to_vec()))
    }

    #[test]
    fn read_yields_a_validated_frame() {
        let frame = reader("CCO\n").read().unwrap().expect("one record");
        assert_eq!(frame["atoms"].n_rows(), Some(3));
        // `read` returns through the schema check, so this frame conforms.
        crate::core::schema::Validator::canonical()
            .validate(&frame)
            .expect("smiles frames conform");
    }

    #[test]
    fn read_as_atomistic_skips_the_frame_round_trip() {
        // The parser produces an Atomistic; asking for one must not go
        // Atomistic -> Frame -> Atomistic. Bonds survive either way, so the
        // observable claim is the count, and the non-round-trip is what the
        // `from_atomistic` override exists for.
        let mol = reader("CCO\n")
            .read_as::<Atomistic>()
            .unwrap()
            .expect("one record");
        assert_eq!(mol.atoms().count(), 3);
        assert_eq!(mol.bonds().count(), 2);
    }

    #[test]
    fn read_as_frame_matches_read() {
        let a = reader("CCO\n").read().unwrap().unwrap();
        let b = reader("CCO\n").read_as::<Frame>().unwrap().unwrap();
        assert_eq!(a["atoms"].n_rows(), b["atoms"].n_rows());
    }

    #[test]
    fn blank_and_comment_lines_are_skipped() {
        let mut r = reader("\n# a comment\nCC\n\n");
        assert!(r.read().unwrap().is_some());
        assert!(r.read().unwrap().is_none());
    }

    #[test]
    fn trailing_name_field_is_ignored() {
        let mol = reader("CCO ethanol\n")
            .read_as::<Atomistic>()
            .unwrap()
            .unwrap();
        assert_eq!(mol.atoms().count(), 3);
    }

    #[test]
    fn descriptor_line_is_refused_and_names_the_fragment_entry_point() {
        // A `.smi` line must never read into an `Atomistic` that silently lost
        // its descriptors; the surfaced message is how the caller finds the
        // fragment entry point.
        let err = reader("[>]COC[<]\n").read().unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("SmilesIr::from_fragment"),
            "got {err}"
        );
    }
}
