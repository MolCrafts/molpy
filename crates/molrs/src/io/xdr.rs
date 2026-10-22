//! Minimal XDR (RFC 4506) primitives for the GROMACS binary trajectory
//! formats (TRR, XTC). All scalars are **big-endian**; opaque/string payloads
//! are zero-padded to a 4-byte boundary. Hand-rolled — no external `xdr`
//! dependency, no `xdrfile` C library.
//!
//! Only the subset the TRR/XTC readers and writers need is implemented:
//! `i32`/`u32`/`f32`/`f64` scalars, opaque byte blocks, and the length-prefixed
//! version string GROMACS stamps into each TRR frame header.

use std::io::{self, Read, Write};

/// Round `n` up to the next multiple of 4 (XDR alignment unit).
#[inline]
pub fn pad4(n: usize) -> usize {
    n.div_ceil(4) * 4
}

/// Number of zero-padding bytes that follow an opaque payload of `n` bytes.
#[inline]
pub fn pad_len(n: usize) -> usize {
    pad4(n) - n
}

/// Read a big-endian `i32`.
#[inline]
pub fn decode_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_be_bytes(b))
}

/// Read a big-endian `u32`.
#[inline]
pub fn decode_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_be_bytes(b))
}

/// Read a big-endian `i64` (two XDR words, high word first).
#[inline]
pub fn decode_i64<R: Read>(r: &mut R) -> io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_be_bytes(b))
}

/// Read a big-endian `f32`.
#[inline]
pub fn decode_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_be_bytes(b))
}

/// Read a big-endian `f64`.
#[inline]
pub fn decode_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_be_bytes(b))
}

/// Read a "real" (single or double precision) as `f64`, selecting width by the
/// `is_double` flag the GROMACS header precision check resolves.
#[inline]
pub fn decode_real<R: Read>(r: &mut R, is_double: bool) -> io::Result<f64> {
    if is_double {
        decode_f64(r)
    } else {
        Ok(decode_f32(r)? as f64)
    }
}

/// Read `n` opaque bytes, then consume the XDR zero-padding up to a 4-byte
/// boundary.
pub fn decode_opaque<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    let pad = pad_len(n);
    if pad > 0 {
        let mut skip = [0u8; 4];
        r.read_exact(&mut skip[..pad])?;
    }
    Ok(buf)
}

/// Read a standard XDR string: a `u32` byte-count prefix, then that many bytes
/// padded to a 4-byte boundary. Returns the decoded text (lossy UTF-8), with
/// any trailing NULs stripped.
///
/// Reading the length dynamically (rather than assuming a fixed version
/// string) keeps the TRR header parser robust across GROMACS versions. Note
/// the TRR frame header precedes this XDR string with a separate `i32`
/// allocation hint (`strlen + 1`); that leading int is consumed by the TRR
/// reader, not here — see [`crate::io::trr::codec`].
pub fn decode_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = decode_u32(r)? as usize;
    let bytes = decode_opaque(r, len)?;
    Ok(String::from_utf8_lossy(&bytes)
        .trim_end_matches('\0')
        .to_string())
}

/// Write a big-endian `i32`.
#[inline]
pub fn encode_i32<W: Write>(w: &mut W, v: i32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

/// Write a big-endian `u32`.
#[inline]
pub fn encode_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

/// Write a big-endian `f32`.
#[inline]
pub fn encode_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

/// Write a big-endian `f64`.
#[inline]
#[cfg(test)]
pub fn encode_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

/// Write `bytes` as an XDR opaque block: the bytes followed by zero-padding to
/// a 4-byte boundary.
pub fn encode_opaque<W: Write>(w: &mut W, bytes: &[u8]) -> io::Result<()> {
    w.write_all(bytes)?;
    let pad = pad_len(bytes.len());
    if pad > 0 {
        w.write_all(&[0u8; 4][..pad])?;
    }
    Ok(())
}

/// Write a standard XDR string: a `u32` byte-count prefix (= `s.len()`, no
/// implicit NUL), then the bytes padded to a 4-byte boundary.
pub fn encode_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    encode_u32(w, bytes.len() as u32)?;
    encode_opaque(w, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn pad4_rounds_up() {
        assert_eq!(pad4(0), 0);
        assert_eq!(pad4(1), 4);
        assert_eq!(pad4(4), 4);
        assert_eq!(pad4(5), 8);
        assert_eq!(pad_len(13), 3);
        assert_eq!(pad_len(12), 0);
    }

    #[test]
    fn scalar_round_trip_big_endian() {
        let mut buf = Vec::new();
        encode_i32(&mut buf, -42).unwrap();
        encode_u32(&mut buf, 1995).unwrap();
        encode_f32(&mut buf, 1.5).unwrap();
        encode_f64(&mut buf, -2.25).unwrap();
        // Big-endian sanity: 1995 = 0x000007CB.
        assert_eq!(&buf[4..8], &[0x00, 0x00, 0x07, 0xCB]);

        let mut c = Cursor::new(buf);
        assert_eq!(decode_i32(&mut c).unwrap(), -42);
        assert_eq!(decode_u32(&mut c).unwrap(), 1995);
        assert_eq!(decode_f32(&mut c).unwrap(), 1.5);
        assert_eq!(decode_f64(&mut c).unwrap(), -2.25);
    }

    #[test]
    fn opaque_round_trip_with_padding() {
        let mut buf = Vec::new();
        encode_opaque(&mut buf, &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(buf.len(), 8, "5 bytes padded to 8");
        assert_eq!(&buf[5..], &[0, 0, 0]);

        let mut c = Cursor::new(buf);
        assert_eq!(decode_opaque(&mut c, 5).unwrap(), vec![1, 2, 3, 4, 5]);
        assert_eq!(c.position(), 8, "padding consumed");
    }

    #[test]
    fn string_round_trip_standard_xdr() {
        let mut buf = Vec::new();
        encode_string(&mut buf, "GMX_trn_file").unwrap();
        // Standard XDR: byte-count prefix = strlen = 12 (no implicit NUL).
        assert_eq!(u32::from_be_bytes(buf[0..4].try_into().unwrap()), 12);
        // 12 bytes already 4-aligned, plus the 4-byte length prefix.
        assert_eq!(buf.len(), 16);

        let mut c = Cursor::new(buf);
        assert_eq!(decode_string(&mut c).unwrap(), "GMX_trn_file");
        assert_eq!(c.position(), 16, "length + padded payload consumed");
    }

    #[test]
    fn decode_real_selects_width() {
        let mut buf = Vec::new();
        encode_f32(&mut buf, 3.0).unwrap();
        encode_f64(&mut buf, 3.0).unwrap();
        let mut c = Cursor::new(buf);
        assert_eq!(decode_real(&mut c, false).unwrap(), 3.0);
        assert_eq!(decode_real(&mut c, true).unwrap(), 3.0);
        assert_eq!(c.position(), 12, "f32 then f64 consumed");
    }
}
