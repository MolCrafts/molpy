//! DCD binary trajectory file reader and writer.

use crate::core::Cell;
use crate::io::frame_index::{FrameIndexBuilder, FrameOffset};
use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::writer::{FrameWriter, Writer};
use crate::io::{Sink, SinkWriter, Source, invalid_data};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::FrameAccess;
use molrs::core::SimBox;
use molrs::op::{F, Idx, Pbc3};
use ndarray::{Array1, IxDyn};
use std::io::{BufRead, Cursor, Read, Seek, SeekFrom, Write};
use std::sync::OnceLock;

// ============================================================================
// Helpers
// ============================================================================

fn unsupported<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, e.to_string())
}

/// Byte order of a DCD file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteOrder {
    /// Little-endian (x86, ARM, NAMD on most platforms).
    Le,
    /// Big-endian (s390x, older PowerPC).
    Be,
}

/// Width of Fortran record-length markers in a DCD file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MarkerSize {
    /// 4-byte markers (default).
    Four,
    /// 8-byte markers (`-frecord-marker=8` builds of CHARMM).
    Eight,
}

impl MarkerSize {
    #[inline]
    fn bytes(self) -> u64 {
        match self {
            MarkerSize::Four => 4,
            MarkerSize::Eight => 8,
        }
    }
}

#[inline]
fn read_u32(buf: &[u8; 4], byte_order: ByteOrder) -> u32 {
    match byte_order {
        ByteOrder::Le => u32::from_le_bytes(*buf),
        ByteOrder::Be => u32::from_be_bytes(*buf),
    }
}

#[inline]
fn read_i32(buf: &[u8; 4], byte_order: ByteOrder) -> i32 {
    match byte_order {
        ByteOrder::Le => i32::from_le_bytes(*buf),
        ByteOrder::Be => i32::from_be_bytes(*buf),
    }
}

#[inline]
fn read_f32(buf: &[u8; 4], byte_order: ByteOrder) -> f32 {
    match byte_order {
        ByteOrder::Le => f32::from_le_bytes(*buf),
        ByteOrder::Be => f32::from_be_bytes(*buf),
    }
}

#[inline]
fn read_f64(buf: &[u8; 8], byte_order: ByteOrder) -> f64 {
    match byte_order {
        ByteOrder::Le => f64::from_le_bytes(*buf),
        ByteOrder::Be => f64::from_be_bytes(*buf),
    }
}

/// Read a Fortran record-length marker at the current reader position.
fn read_marker<R: Read>(
    reader: &mut R,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
) -> std::io::Result<u64> {
    match marker_size {
        MarkerSize::Four => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(read_u32(&buf, byte_order) as u64)
        }
        MarkerSize::Eight => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            let v = match byte_order {
                ByteOrder::Le => u64::from_le_bytes(buf),
                ByteOrder::Be => u64::from_be_bytes(buf),
            };
            Ok(v)
        }
    }
}

/// Write a Fortran record-length marker.
fn write_marker<W: Write>(
    writer: &mut W,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
    value: u64,
) -> std::io::Result<()> {
    match marker_size {
        MarkerSize::Four => {
            let v = u32::try_from(value).map_err(invalid_data)?;
            let bytes = match byte_order {
                ByteOrder::Le => v.to_le_bytes(),
                ByteOrder::Be => v.to_be_bytes(),
            };
            writer.write_all(&bytes)
        }
        MarkerSize::Eight => {
            let bytes = match byte_order {
                ByteOrder::Le => value.to_le_bytes(),
                ByteOrder::Be => value.to_be_bytes(),
            };
            writer.write_all(&bytes)
        }
    }
}

/// Refuse a Fortran payload bigger than this before allocating, so a corrupt
/// marker on a huge file cannot request a multi-gigabyte `Vec`.
const MAX_DCD_RECORD_BYTES: u64 = 256 * 1024 * 1024;

/// Read one Fortran record's payload, asserting the trailing marker matches.
fn read_record<R: Read>(
    reader: &mut R,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    read_record_into_max(
        reader,
        byte_order,
        marker_size,
        &mut buf,
        MAX_DCD_RECORD_BYTES,
    )?;
    Ok(buf)
}

fn read_record_into_max<R: Read>(
    reader: &mut R,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
    buf: &mut Vec<u8>,
    max_payload: u64,
) -> std::io::Result<()> {
    let leading = read_marker(reader, byte_order, marker_size)?;
    if leading > max_payload {
        return Err(invalid_data(format!(
            "Fortran record payload {leading} bytes exceeds cap {max_payload}"
        )));
    }
    let n = usize::try_from(leading).map_err(invalid_data)?;
    buf.resize(n, 0);
    reader.read_exact(buf)?;
    let trailing = read_marker(reader, byte_order, marker_size)?;
    if leading != trailing {
        return Err(invalid_data(format!(
            "Fortran record marker mismatch: leading={}, trailing={}",
            leading, trailing
        )));
    }
    Ok(())
}

// ============================================================================
// Header
// ============================================================================

/// Parsed DCD header.
#[derive(Debug, Clone)]
pub(crate) struct DcdHeader {
    /// Byte order of the file.
    pub byte_order: ByteOrder,
    /// Width of Fortran record-length markers.
    pub marker_size: MarkerSize,
    /// CHARMM version field; 0 indicates an X-PLOR-style header.
    pub charmm_ver: i32,
    /// Number of frames as recorded in the header (NSET).
    pub nset: u32,
    /// Starting timestep (ISTART).
    pub istart: i32,
    /// Number of integration steps between saved frames (NSAVC).
    pub nsavc: i32,
    /// Total number of atoms (NATOMS).
    pub natoms: u32,
    /// Number of fixed atoms (NAMNF).
    pub namnf: u32,
    /// Whether per-frame box records are present.
    pub has_box: bool,
    /// Whether per-frame W coordinates are present (4D dynamics).
    pub has_4d: bool,
    /// Integration timestep in AKMA units (DELTA).
    pub delta: f64,
    /// Concatenated title strings, trimmed.
    pub title: String,
    /// 1-indexed free-atom indices (only present when NAMNF > 0).
    pub free_atoms: Option<Vec<i32>>,
    /// Frame-0 X/Y/Z coords, cached when NAMNF > 0 so later frames can be
    /// reconstructed by overlaying the free-atom subset on top of these.
    fixed_seed: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    /// Byte offset of the first per-frame record.
    pub data_offset: u64,
    /// Byte size of frame 0 (full NATOMS).
    pub frame_size_first: u64,
    /// Byte size of frames 1..NSET (NATOMS - NAMNF when NAMNF > 0; otherwise
    /// equal to `frame_size_first`).
    pub frame_size_rest: u64,
}

impl DcdHeader {
    /// Byte offset of frame `n` in the file.
    fn frame_offset(&self, n: usize) -> u64 {
        if n == 0 {
            self.data_offset
        } else {
            self.data_offset + self.frame_size_first + (n as u64 - 1) * self.frame_size_rest
        }
    }

    /// Number of effective coordinate values in frame `n`.
    fn natoms_eff(&self, n: usize) -> u32 {
        if self.namnf == 0 || n == 0 {
            self.natoms
        } else {
            self.natoms - self.namnf
        }
    }
}

/// Header fields that do not need the file size / first-frame probe.
#[derive(Debug, Clone)]
struct HeaderPartial {
    byte_order: ByteOrder,
    marker_size: MarkerSize,
    charmm_ver: i32,
    istart: i32,
    nsavc: i32,
    natoms: u32,
    namnf: u32,
    delta: f64,
    /// `(has_box, has_4d)` as a CHARMM-flavour header states them
    /// (`icntrl[10]`, `icntrl[11]`); `None` for an X-PLOR header, whose
    /// layout is inferred from the record sizes.
    charmm_flags: Option<(bool, bool)>,
    title: String,
    free_atoms: Option<Vec<i32>>,
    data_offset: u64,
}

#[derive(Debug, Clone, Copy)]
struct FrameLayout {
    has_box: bool,
    has_4d: bool,
    frame_size_first: u64,
    frame_size_rest: u64,
    nset: u32,
}

impl HeaderPartial {
    fn into_header(self, layout: FrameLayout) -> DcdHeader {
        DcdHeader {
            byte_order: self.byte_order,
            marker_size: self.marker_size,
            charmm_ver: self.charmm_ver,
            nset: layout.nset,
            istart: self.istart,
            nsavc: self.nsavc,
            natoms: self.natoms,
            namnf: self.namnf,
            has_box: layout.has_box,
            has_4d: layout.has_4d,
            delta: self.delta,
            title: self.title,
            free_atoms: self.free_atoms,
            fixed_seed: None,
            data_offset: self.data_offset,
            frame_size_first: layout.frame_size_first,
            frame_size_rest: layout.frame_size_rest,
        }
    }

    fn empty_layout() -> FrameLayout {
        FrameLayout {
            has_box: false,
            has_4d: false,
            frame_size_first: 0,
            frame_size_rest: 0,
            nset: 0,
        }
    }
}

/// Detect byte order and Fortran marker width.
///
/// The first record always wraps the 84-byte CORD header. The marker layout
/// is ambiguous when read in isolation (`84u64.to_le_bytes()` and
/// `84u32.to_le_bytes() ++ [0u8; 4]` are byte-for-byte identical), so we
/// disambiguate by checking which marker width places the ASCII `"CORD"`
/// magic immediately after the leading marker.
fn detect_endianness_and_marker<R: Read>(
    reader: &mut R,
) -> std::io::Result<(ByteOrder, MarkerSize)> {
    let mut buf = [0u8; 12];
    reader.read_exact(&mut buf)?;

    // 4-byte marker: leading marker at [0..4], CORD at [4..8].
    if &buf[4..8] == b"CORD" {
        let leading: [u8; 4] = buf[..4].try_into().unwrap();
        if u32::from_le_bytes(leading) == 84 {
            return Ok((ByteOrder::Le, MarkerSize::Four));
        }
        if u32::from_be_bytes(leading) == 84 {
            return Ok((ByteOrder::Be, MarkerSize::Four));
        }
    }

    // 8-byte marker: leading marker at [0..8], CORD at [8..12].
    if &buf[8..12] == b"CORD" {
        let leading: [u8; 8] = buf[..8].try_into().unwrap();
        if u64::from_le_bytes(leading) == 84 {
            return Ok((ByteOrder::Le, MarkerSize::Eight));
        }
        if u64::from_be_bytes(leading) == 84 {
            return Ok((ByteOrder::Be, MarkerSize::Eight));
        }
    }

    Err(invalid_data(format!(
        "not a DCD file: first 12 bytes {:02x?} do not match any known marker layout",
        buf
    )))
}

/// Parse the CORD / title / NATOMS / optional free-atom records. Leaves the
/// reader at `data_offset`. Does **not** probe file size or per-frame layout.
fn parse_header_records<R: Read + Seek>(reader: &mut R) -> std::io::Result<HeaderPartial> {
    reader.seek(SeekFrom::Start(0))?;
    let (byte_order, marker_size) = detect_endianness_and_marker(reader)?;

    // Re-read from the start now that we know the layout.
    reader.seek(SeekFrom::Start(0))?;

    // -- Header record 1 (84 bytes) --
    let h1 = read_record(reader, byte_order, marker_size)?;
    if h1.len() != 84 {
        return Err(invalid_data(format!(
            "header record 1 has {} bytes, expected 84",
            h1.len()
        )));
    }
    if &h1[0..4] != b"CORD" {
        return Err(invalid_data(format!(
            "header magic is {:?}, expected 'CORD'",
            std::str::from_utf8(&h1[0..4]).unwrap_or("?")
        )));
    }

    let read_i32_at = |off: usize| -> i32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&h1[off..off + 4]);
        read_i32(&buf, byte_order)
    };

    let istart = read_i32_at(8);
    let nsavc = read_i32_at(12);
    let namnf = read_i32_at(36) as u32;
    let charmm_ver = read_i32_at(80);

    // DELTA position depends on header flavor: CHARMM stores f32 at icntrl[9]
    // (payload offset 40); X-PLOR stores f64 spanning icntrl[9..10] (payload
    // offsets 40..48). icntrl[19] (payload offset 80) chooses the flavor.
    let delta = if charmm_ver != 0 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&h1[40..44]);
        read_f32(&buf, byte_order) as f64
    } else {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&h1[40..48]);
        read_f64(&buf, byte_order)
    };
    // CHARMM: the crystal flag at icntrl[10], the 4D flag at icntrl[11].
    let charmm_flags = (charmm_ver != 0).then(|| (read_i32_at(44) != 0, read_i32_at(48) != 0));

    // -- Header record 2 (title) --
    let h2 = read_record(reader, byte_order, marker_size)?;
    if h2.len() < 4 {
        return Err(invalid_data("title record too short"));
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&h2[0..4]);
    let ntitle = read_i32(&buf, byte_order);
    if ntitle < 0 {
        return Err(invalid_data(format!("invalid NTITLE={}", ntitle)));
    }
    let expected_title_len = 4 + (ntitle as usize) * 80;
    if h2.len() < expected_title_len {
        return Err(invalid_data(format!(
            "title record has {} bytes, expected at least {}",
            h2.len(),
            expected_title_len
        )));
    }
    let mut title = String::with_capacity((ntitle as usize) * 80);
    for i in 0..ntitle as usize {
        let off = 4 + i * 80;
        let s = String::from_utf8_lossy(&h2[off..off + 80]);
        title.push_str(s.trim_end_matches([' ', '\0']));
        title.push('\n');
    }
    let title = title.trim_end_matches('\n').to_owned();

    // -- Header record 3 (NATOMS) --
    let h3 = read_record(reader, byte_order, marker_size)?;
    if h3.len() != 4 {
        return Err(invalid_data(format!(
            "NATOMS record has {} bytes, expected 4",
            h3.len()
        )));
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&h3[0..4]);
    let natoms = read_i32(&buf, byte_order);
    if natoms <= 0 {
        return Err(invalid_data(format!("invalid NATOMS={}", natoms)));
    }
    let natoms = natoms as u32;

    if namnf > natoms {
        return Err(invalid_data(format!(
            "NAMNF={} exceeds NATOMS={}",
            namnf, natoms
        )));
    }

    // -- Optional fixed-atom record --
    let free_atoms = if namnf > 0 {
        let payload = read_record(reader, byte_order, marker_size)?;
        let nfree = (natoms - namnf) as usize;
        if payload.len() != nfree * 4 {
            return Err(invalid_data(format!(
                "fixed-atom record has {} bytes, expected {}",
                payload.len(),
                nfree * 4
            )));
        }
        let mut indices = Vec::with_capacity(nfree);
        for i in 0..nfree {
            let mut b = [0u8; 4];
            b.copy_from_slice(&payload[i * 4..(i + 1) * 4]);
            indices.push(read_i32(&b, byte_order));
        }
        Some(indices)
    } else {
        None
    };

    let data_offset = reader.stream_position()?;
    Ok(HeaderPartial {
        byte_order,
        marker_size,
        charmm_ver,
        istart,
        nsavc,
        natoms,
        namnf,
        delta,
        charmm_flags,
        title,
        free_atoms,
        data_offset,
    })
}

/// The per-frame byte sizes `(first, rest)` the header flags state, once the
/// first record agrees with them.
///
/// # Errors
///
/// A first record whose size contradicts the crystal flag.
fn layout_from_flags(
    partial: &HeaderPartial,
    first_marker: u64,
    has_box: bool,
    has_4d: bool,
) -> std::io::Result<(u64, u64)> {
    let m = partial.marker_size.bytes();
    let coord_marker_full = u64::from(partial.natoms) * 4;
    let predicted_first = if has_box { 48 } else { coord_marker_full };
    if first_marker != predicted_first {
        return Err(invalid_data(format!(
            "the header says the file {} a unit cell, but the first record is {first_marker} \
             bytes, not {predicted_first}",
            if has_box { "has" } else { "has no" }
        )));
    }
    let box_part = if has_box { 2 * m + 48 } else { 0 };
    let n_recs = if has_4d { 4 } else { 3 };
    let coord_rec_full = 2 * m + coord_marker_full;
    let coord_rec_eff = 2 * m + u64::from(partial.natoms - partial.namnf) * 4;
    Ok((
        box_part + coord_rec_full * n_recs,
        box_part + coord_rec_eff * n_recs,
    ))
}

fn try_parse_header_records(bytes: &[u8]) -> std::io::Result<Option<HeaderPartial>> {
    let mut cursor = Cursor::new(bytes);
    match parse_header_records(&mut cursor) {
        Ok(partial) => Ok(Some(partial)),
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(err) => Err(err),
    }
}

/// Pick `(has_box, has_4d)` from the first per-frame marker and the length
/// of the data section. Same candidate walk the seek-based reader uses.
fn resolve_frame_layout(
    partial: &HeaderPartial,
    first_marker: u64,
    trailing: u64,
) -> std::io::Result<FrameLayout> {
    if trailing == 0 {
        return Ok(HeaderPartial::empty_layout());
    }

    // A CHARMM header states the layout; only an X-PLOR header is inferred.
    if let Some((has_box, has_4d)) = partial.charmm_flags {
        let (frame_size_first, frame_size_rest) =
            layout_from_flags(partial, first_marker, has_box, has_4d)?;
        if trailing < frame_size_first {
            return Err(invalid_data(format!(
                "the data section holds {trailing} bytes, less than one {frame_size_first}-byte \
                 frame"
            )));
        }
        let nset = 1 + (trailing - frame_size_first) / frame_size_rest;
        return Ok(FrameLayout {
            has_box,
            has_4d,
            frame_size_first,
            frame_size_rest,
            nset: u32::try_from(nset).unwrap_or(u32::MAX),
        });
    }

    let m = partial.marker_size.bytes();
    let natoms = partial.natoms;
    let namnf = partial.namnf;
    let coord_marker_full = u64::from(natoms) * 4;
    let coord_rec_full = 2 * m + coord_marker_full;
    let coord_rec_eff = 2 * m + u64::from(natoms - namnf) * 4;

    // (has_box, has_4d, f_first, f_rest, missing_bytes_of_next_frame)
    // missing == 0 → file ends on a frame boundary. Otherwise the last
    // frame is truncated (killed dump, incomplete copy). Do not reject the
    // whole file: keep every complete frame.
    let mut candidates: Vec<(bool, bool, u64, u64, u64)> = Vec::new();
    for &has_box_candidate in &[true, false] {
        let box_part = if has_box_candidate { 2 * m + 48 } else { 0 };
        let predicted_first = if has_box_candidate {
            48
        } else {
            coord_marker_full
        };
        if first_marker != predicted_first {
            continue;
        }
        for &has_4d_candidate in &[false, true] {
            let n_recs = if has_4d_candidate { 4 } else { 3 };
            let f_first = box_part + coord_rec_full * n_recs;
            let f_rest = box_part + coord_rec_eff * n_recs;
            if f_first == 0 || f_rest == 0 || trailing < f_first {
                continue;
            }
            let after_first = trailing - f_first;
            let rem = after_first % f_rest;
            let missing = if rem == 0 { 0 } else { f_rest - rem };
            candidates.push((
                has_box_candidate,
                has_4d_candidate,
                f_first,
                f_rest,
                missing,
            ));
        }
    }
    if candidates.is_empty() {
        return Err(invalid_data(format!(
            "no per-frame layout (has_box, has_4d) is consistent with file size {} after header (natoms={}, namnf={}, first_marker={})",
            trailing, natoms, namnf, first_marker
        )));
    }
    // Exact end-of-file first. Then the layout whose leftover is closest to
    // a truncated next frame (smallest missing). Tie-break: no 4D, then no box.
    candidates.sort_by_key(|&(hb, h4, _, _, missing)| (missing, h4 as u8, hb as u8));
    let (has_box, has_4d, frame_size_first, frame_size_rest, _) = candidates[0];
    let actual_nset = 1 + (trailing - frame_size_first) / frame_size_rest;
    Ok(FrameLayout {
        has_box,
        has_4d,
        frame_size_first,
        frame_size_rest,
        nset: u32::try_from(actual_nset).unwrap_or(u32::MAX),
    })
}

/// Parse the full DCD header from the current reader position. Advances the
/// reader past the header (and, when NAMNF > 0, past frame 0 — frame 0's
/// coordinates are cached in the returned header).
fn parse_header<R: BufRead + Seek>(reader: &mut R) -> std::io::Result<DcdHeader> {
    let partial = parse_header_records(reader)?;
    let data_offset = partial.data_offset;
    let end = reader.seek(SeekFrom::End(0))?;
    let trailing = end.saturating_sub(data_offset);

    if trailing == 0 {
        return Ok(partial.into_header(HeaderPartial::empty_layout()));
    }

    reader.seek(SeekFrom::Start(data_offset))?;
    let first_marker = read_marker(reader, partial.byte_order, partial.marker_size)?;
    reader.seek(SeekFrom::Start(data_offset))?;

    let layout = resolve_frame_layout(&partial, first_marker, trailing)?;
    let mut header = partial.into_header(layout);

    // Cache frame-0 coordinates if we need them to reconstruct fixed atoms.
    if header.namnf > 0 && header.nset > 0 {
        let (xs, ys, zs, _w) = read_coord_payload(reader, &header, 0)?;
        header.fixed_seed = Some((xs, ys, zs));
        // Reset; read_frame will re-read frame 0 if requested.
        reader.seek(SeekFrom::Start(data_offset))?;
    }

    Ok(header)
}

// ============================================================================
// Frame parsing
// ============================================================================

/// Read the coordinate (and box) payload for frame `n`. Reader must be
/// positioned at the start of frame `n`. Returns `(x, y, z, w)` where `w` is
/// `Some` iff `header.has_4d`.
///
/// For fixed-atom files with `n >= 1`, the returned arrays still have length
/// `natoms`: fixed atoms are filled from `header.fixed_seed`.
type Coords = (Vec<f32>, Vec<f32>, Vec<f32>, Option<Vec<f32>>);

fn read_coord_payload<R: Read>(
    reader: &mut R,
    header: &DcdHeader,
    n: usize,
) -> std::io::Result<Coords> {
    let natoms = header.natoms as usize;
    let natoms_eff = header.natoms_eff(n) as usize;

    // Skip past the box record — caller handles it separately when needed.
    if header.has_box {
        let _ = read_record(reader, header.byte_order, header.marker_size)?;
    }

    let mut buf = Vec::with_capacity(natoms_eff * 4);
    let axis_bytes = (natoms_eff as u64).saturating_mul(4);

    let read_axis = |reader: &mut R, buf: &mut Vec<u8>| -> std::io::Result<Vec<f32>> {
        read_record_into_max(
            reader,
            header.byte_order,
            header.marker_size,
            buf,
            axis_bytes,
        )?;
        if buf.len() != natoms_eff * 4 {
            return Err(invalid_data(format!(
                "coord record has {} bytes, expected {}",
                buf.len(),
                natoms_eff * 4
            )));
        }
        let mut out = vec![0.0f32; natoms_eff];
        for i in 0..natoms_eff {
            let mut b = [0u8; 4];
            b.copy_from_slice(&buf[i * 4..(i + 1) * 4]);
            out[i] = read_f32(&b, header.byte_order);
        }
        Ok(out)
    };

    let xs_eff = read_axis(reader, &mut buf)?;
    let ys_eff = read_axis(reader, &mut buf)?;
    let zs_eff = read_axis(reader, &mut buf)?;
    let w_eff = if header.has_4d {
        Some(read_axis(reader, &mut buf)?)
    } else {
        None
    };

    if header.namnf == 0 || n == 0 {
        return Ok((xs_eff, ys_eff, zs_eff, w_eff));
    }

    // Reconstruct full-length arrays by overlaying free atoms onto the cached
    // frame-0 fixed coords.
    let (sx, sy, sz) = header
        .fixed_seed
        .as_ref()
        .ok_or_else(|| invalid_data("fixed_seed missing for fixed-atom DCD"))?;
    let free = header
        .free_atoms
        .as_ref()
        .ok_or_else(|| invalid_data("free_atoms missing for fixed-atom DCD"))?;
    if free.len() != natoms_eff {
        return Err(invalid_data(format!(
            "free_atoms has {} entries, expected {}",
            free.len(),
            natoms_eff
        )));
    }

    let mut xs = sx.clone();
    let mut ys = sy.clone();
    let mut zs = sz.clone();
    for (slot, &one_indexed) in free.iter().enumerate() {
        if one_indexed < 1 || (one_indexed as usize) > natoms {
            return Err(invalid_data(format!(
                "free atom index {} out of range 1..={}",
                one_indexed, natoms
            )));
        }
        let i = (one_indexed - 1) as usize;
        xs[i] = xs_eff[slot];
        ys[i] = ys_eff[slot];
        zs[i] = zs_eff[slot];
    }
    // 4D w is not propagated through the seed (no test corpus exercises it).
    Ok((xs, ys, zs, w_eff))
}

/// Parse the box record at the current reader position into a `SimBox`.
fn parse_box_payload(payload: &[u8], byte_order: ByteOrder) -> std::io::Result<SimBox> {
    if payload.len() != 48 {
        return Err(invalid_data(format!(
            "box record has {} bytes, expected 48",
            payload.len()
        )));
    }
    let read_d = |off: usize| -> f64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&payload[off..off + 8]);
        read_f64(&b, byte_order)
    };

    // CHARMM order: A, gamma, B, beta, alpha, C
    let a = read_d(0);
    let raw_gamma = read_d(8);
    let b = read_d(16);
    let raw_beta = read_d(24);
    let raw_alpha = read_d(32);
    let c = read_d(40);

    // Cosine-vs-degree heuristic: NAMD/CHARMM ≥ 25 stores cosines, older
    // CHARMM stores degrees. If all three angle slots fit inside [-1, 1] we
    // treat them as cosines.
    let in_unit = |v: f64| (-1.0..=1.0).contains(&v);
    let (alpha, beta, gamma) = if in_unit(raw_alpha) && in_unit(raw_beta) && in_unit(raw_gamma) {
        (
            raw_alpha.acos().to_degrees(),
            raw_beta.acos().to_degrees(),
            raw_gamma.acos().to_degrees(),
        )
    } else {
        (raw_alpha, raw_beta, raw_gamma)
    };

    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return Err(invalid_data(format!(
            "non-positive box length: A={}, B={}, C={}",
            a, b, c
        )));
    }

    let pbc: Pbc3 = [true; 3];
    let origin = [0.0; 3];

    if (alpha - 90.0).abs() < 1e-6 && (beta - 90.0).abs() < 1e-6 && (gamma - 90.0).abs() < 1e-6 {
        return SimBox::new(Cell::Lengths([a as F, b as F, c as F]), origin, pbc)
            .map_err(|e| invalid_data(format!("ortho box: {:?}", e)));
    }

    let cell = Cell::LengthsAngles {
        lengths: [a, b, c],
        angles: [alpha, beta, gamma],
    };
    SimBox::new(cell, origin, pbc).map_err(|e| invalid_data(format!("triclinic box: {:?}", e)))
}

/// Decode one frame whose first byte is at the current reader position.
fn parse_one_frame<R: Read>(
    reader: &mut R,
    header: &DcdHeader,
    n: usize,
) -> std::io::Result<Frame> {
    let simbox = if header.has_box {
        let mut payload = Vec::new();
        read_record_into_max(
            reader,
            header.byte_order,
            header.marker_size,
            &mut payload,
            48,
        )?;
        Some(parse_box_payload(&payload, header.byte_order)?)
    } else {
        None
    };

    let mut coords_header = header.clone();
    coords_header.has_box = false;
    let (xs, ys, zs, w) = read_coord_payload(reader, &coords_header, n)?;

    let natoms = header.natoms as usize;
    let mut atoms = Block::new();
    let id_arr = Array1::from_iter(1..=natoms as Idx)
        .into_shape_with_order(IxDyn(&[natoms]))
        .map_err(invalid_data)?;
    atoms.insert("id", id_arr, None).map_err(invalid_data)?;

    let to_f64 = |v: &[f32]| -> Array1<F> { v.iter().map(|&x| x as F).collect() };

    atoms
        .insert(
            "x",
            to_f64(&xs)
                .into_shape_with_order(IxDyn(&[natoms]))
                .map_err(invalid_data)?,
            None,
        )
        .map_err(invalid_data)?;
    atoms
        .insert(
            "y",
            to_f64(&ys)
                .into_shape_with_order(IxDyn(&[natoms]))
                .map_err(invalid_data)?,
            None,
        )
        .map_err(invalid_data)?;
    atoms
        .insert(
            "z",
            to_f64(&zs)
                .into_shape_with_order(IxDyn(&[natoms]))
                .map_err(invalid_data)?,
            None,
        )
        .map_err(invalid_data)?;
    if let Some(w) = w {
        atoms
            .insert(
                "w",
                to_f64(&w)
                    .into_shape_with_order(IxDyn(&[natoms]))
                    .map_err(invalid_data)?,
                None,
            )
            .map_err(invalid_data)?;
    }

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.simbox = simbox;

    let timestep = (header.istart as i64) + (n as i64) * (header.nsavc as i64);
    frame
        .meta
        .insert("timestep".to_string(), timestep.to_string());
    frame
        .meta
        .insert("delta".to_string(), header.delta.to_string());
    if !header.title.is_empty() {
        frame.meta.insert("title".to_string(), header.title.clone());
    }

    // Every door reads through here, so every door checks here.
    crate::io::reader::check_read_frame(&frame)?;
    Ok(frame)
}

/// Read frame `n` at its computed offset. Reader must be Seek.
fn parse_frame_at<R: BufRead + Seek>(
    reader: &mut R,
    header: &DcdHeader,
    n: usize,
) -> std::io::Result<Option<Frame>> {
    if n >= header.nset as usize {
        return Ok(None);
    }
    reader.seek(SeekFrom::Start(header.frame_offset(n)))?;
    Ok(Some(parse_one_frame(reader, header, n)?))
}

// ============================================================================
// Reader
// ============================================================================

/// DCD trajectory reader implementing `TrajectoryReader` for random access.
///
/// Header parsing is lazy — performed on the first call to a method that
/// needs it. This keeps `Reader::new` infallible.
///
/// DCD is the de-facto binary trajectory format used by CHARMM, NAMD,
/// OpenMM, and (optionally) LAMMPS. The on-disk layout is a sequence of
/// Fortran "unformatted" records: each record is `[len][payload][len]`
/// with two equal length markers framing the payload. Marker width is 4
/// bytes by default and 8 bytes when CHARMM was compiled with
/// `-frecord-marker=8`. Both little- and big-endian byte orders are
/// supported.
///
/// # Supported features
///
/// - 32- and 64-bit Fortran record markers
/// - Little- and big-endian byte order
/// - Optional periodic-box record (NAMD/CHARMM "extra block")
/// - Triclinic boxes encoded as either cosines or degrees
/// - Optional 4D dynamics (W coordinate)
/// - Fixed-atom subsets (frame 0 carries full coords; later frames only
///   the free atoms)
/// - O(1) random access via `TrajectoryReader::read_frame` (frame size is
///   constant, no scan needed)
/// - Writer (NAMD-style: little-endian, 4-byte markers, cosine angles,
///   no fixed atoms, no 4D)
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use molrs::io::dcd::{DcdReader, DcdWriter};
/// use molrs::io::reader::TrajectoryReader;
/// use molrs::io::writer::FrameWriter;
///
/// # fn main() -> std::io::Result<()> {
/// // Random access via TrajectoryReader
/// let mut reader = DcdReader::open(Path::new("trajectory.dcd"))?;
/// let frame_5 = reader.read_frame(5)?;
///
/// // Copy every frame into a new file
/// let mut writer = DcdWriter::create(Path::new("output.dcd"))?;
/// for frame in reader.iter() {
///     writer.write(&frame?)?;
/// }
/// # Ok(())
/// # }
/// ```
pub struct DcdReader<R: BufRead + Seek> {
    reader: R,
    header: OnceLock<DcdHeader>,
    cursor: usize,
}

impl<R: BufRead + Seek> DcdReader<R> {
    /// Wrap `reader` in a `DcdReader`. Header parsing is deferred.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            header: OnceLock::new(),
            cursor: 0,
        }
    }

    fn ensure_header(&mut self) -> std::io::Result<()> {
        if self.header.get().is_some() {
            return Ok(());
        }
        let header = parse_header(&mut self.reader)?;
        self.header
            .set(header)
            .map_err(|_| std::io::Error::other("failed to set header"))?;
        Ok(())
    }
}

impl<R: BufRead + Seek> Reader for DcdReader<R> {
    type R = R;

    fn new(reader: Self::R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for DcdReader<R> {
    fn read(&mut self) -> std::io::Result<Option<Frame>> {
        self.ensure_header()?;
        let cursor = self.cursor;
        let header = self.header.get().expect("header set");
        if cursor >= header.nset as usize {
            return Ok(None);
        }
        let header = header.clone();
        let frame = parse_frame_at(&mut self.reader, &header, cursor)?;
        if frame.is_some() {
            self.cursor += 1;
        }
        Ok(frame)
    }
}

impl<R: BufRead + Seek> TrajectoryReader for DcdReader<R> {
    fn build_index(&mut self) -> std::io::Result<()> {
        self.ensure_header()
    }

    fn read_frame(&mut self, index: usize) -> std::io::Result<Option<Frame>> {
        self.ensure_header()?;
        let header = self.header.get().expect("header set").clone();
        parse_frame_at(&mut self.reader, &header, index)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        self.ensure_header()?;
        Ok(self.header.get().expect("header set").nset as usize)
    }
}

// ============================================================================
// Writer
// ============================================================================

const WRITER_TITLE_DEFAULT: &str = "Created by molcrafts-molrs";

struct WriterMeta {
    natoms: u32,
    has_box: bool,
    nset_offset: u64,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
}

/// DCD trajectory writer.
///
/// Writes NAMD-compatible files: little-endian, 4-byte Fortran markers,
/// CHARMM v24 header, cosine-encoded box angles. Does not support writing
/// 4D dynamics or fixed atoms — both produce `ErrorKind::Unsupported`.
///
/// Requires `Seek` so the NSET field can be patched after each frame.
pub struct DcdWriter<W: Write + Seek> {
    writer: W,
    meta: Option<WriterMeta>,
    nset: u32,
}

impl<W: Write + Seek> DcdWriter<W> {
    /// Create a new DCD writer. The header is emitted on the first
    /// [`FrameWriter::write`](crate::io::writer::FrameWriter::write) call so it can match the frame's atom count and box
    /// presence.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            meta: None,
            nset: 0,
        }
    }
}

impl<W: Write + Seek> Writer for DcdWriter<W> {
    type W = W;

    fn new(writer: Self::W) -> Self {
        Self::new(writer)
    }
}

impl<W: Write + Seek> FrameWriter for DcdWriter<W> {
    fn write(&mut self, frame: &Frame) -> std::io::Result<()> {
        write_dcd_frame(self, frame)?;
        self.writer.flush()
    }
}

fn write_dcd_frame<W: Write + Seek>(
    state: &mut DcdWriter<W>,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    // Every door writes through here, so every door checks here.
    crate::io::writer::check_write_frame(frame)?;
    crate::io::null_cells::refuse_null_cells(frame, "atoms", &["x", "y", "z", "w"])?;
    // Reject 4D / fixed-atom frames eagerly.
    let has_w = frame
        .visit_block("atoms", |a| {
            a.column("w").and_then(|c| c.get::<f64>()).is_some()
        })
        .unwrap_or(false);
    if has_w {
        return Err(unsupported(
            "DcdWriter does not support 4D (w-column) frames",
        ));
    }

    let natoms_in_frame = frame
        .visit_block("atoms", |a| a.n_rows().unwrap_or(0))
        .ok_or_else(|| invalid_data("frame must contain 'atoms' block"))?
        as u32;
    if natoms_in_frame == 0 {
        return Err(invalid_data("frame has no atoms"));
    }

    let has_box = frame.simbox_ref().is_some();

    if state.meta.is_none() {
        let title = frame
            .meta_ref()
            .get("title")
            .and_then(|value| value.as_str())
            .unwrap_or(WRITER_TITLE_DEFAULT);
        let istart: i32 = frame
            .meta_ref()
            .get("timestep")
            .and_then(|value| value.as_i32())
            .unwrap_or(0);
        let nset_offset =
            write_header_stub(&mut state.writer, natoms_in_frame, has_box, title, istart)?;
        state.meta = Some(WriterMeta {
            natoms: natoms_in_frame,
            has_box,
            nset_offset,
            byte_order: ByteOrder::Le,
            marker_size: MarkerSize::Four,
        });
    }
    let meta = state.meta.as_ref().expect("meta set");

    if natoms_in_frame != meta.natoms {
        return Err(invalid_data(format!(
            "DCD requires constant atom count: frame has {}, header has {}",
            natoms_in_frame, meta.natoms
        )));
    }
    if has_box != meta.has_box {
        return Err(invalid_data(
            "DCD requires consistent box presence across frames",
        ));
    }

    write_frame_payload(&mut state.writer, meta, frame)?;
    state.nset = state.nset.saturating_add(1);

    // Patch NSET in place.
    let end_pos = state.writer.stream_position()?;
    state.writer.seek(SeekFrom::Start(meta.nset_offset))?;
    let bytes = state.nset.to_le_bytes();
    state.writer.write_all(&bytes)?;
    state.writer.seek(SeekFrom::Start(end_pos))?;
    Ok(())
}

/// Write a stub header. Returns the byte offset of the NSET field so it can
/// be patched as frames are appended.
fn write_header_stub<W: Write + Seek>(
    writer: &mut W,
    natoms: u32,
    has_box: bool,
    title: &str,
    istart: i32,
) -> std::io::Result<u64> {
    let byte_order = ByteOrder::Le;
    let marker_size = MarkerSize::Four;

    // -- Header record 1 (84 bytes) --
    write_marker(writer, byte_order, marker_size, 84)?;
    writer.write_all(b"CORD")?;

    // Remember NSET position for later patching.
    let nset_offset = writer.stream_position()?;
    writer.write_all(&0u32.to_le_bytes())?; // NSET (patched later)
    writer.write_all(&istart.to_le_bytes())?;
    writer.write_all(&1i32.to_le_bytes())?; // NSAVC
    for _ in 0..5 {
        writer.write_all(&0i32.to_le_bytes())?;
    }
    writer.write_all(&0i32.to_le_bytes())?; // NAMNF
    // The CHARMM words, where the reader decodes them: DELTA at icntrl[9],
    // the crystal flag at icntrl[10], the 4D flag at icntrl[11].
    writer.write_all(&1.0f32.to_le_bytes())?; // DELTA
    writer.write_all(&(if has_box { 1i32 } else { 0i32 }).to_le_bytes())?; // crystal flag
    writer.write_all(&0i32.to_le_bytes())?; // 4D flag
    for _ in 0..7 {
        writer.write_all(&0i32.to_le_bytes())?; // icntrl[12..=18]
    }
    writer.write_all(&24i32.to_le_bytes())?; // CHARMM_VER (NAMD-compatible)
    write_marker(writer, byte_order, marker_size, 84)?;

    // -- Header record 2 (title) --
    let title_lines: Vec<String> = if title.is_empty() {
        vec![WRITER_TITLE_DEFAULT.to_string()]
    } else {
        title.lines().map(|l| l.to_string()).collect()
    };
    let ntitle = title_lines.len() as i32;
    let payload_len = 4 + (ntitle as u64) * 80;
    write_marker(writer, byte_order, marker_size, payload_len)?;
    writer.write_all(&ntitle.to_le_bytes())?;
    for line in &title_lines {
        let mut padded = [b' '; 80];
        let bytes = line.as_bytes();
        let n = bytes.len().min(80);
        padded[..n].copy_from_slice(&bytes[..n]);
        writer.write_all(&padded)?;
    }
    write_marker(writer, byte_order, marker_size, payload_len)?;

    // -- Header record 3 (NATOMS) --
    write_marker(writer, byte_order, marker_size, 4)?;
    writer.write_all(&natoms.to_le_bytes())?;
    write_marker(writer, byte_order, marker_size, 4)?;

    Ok(nset_offset)
}

/// Write the box record + X/Y/Z payload for one frame.
fn write_frame_payload<W: Write>(
    writer: &mut W,
    meta: &WriterMeta,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    if meta.has_box {
        let simbox = frame
            .simbox_ref()
            .ok_or_else(|| invalid_data("frame missing simbox but header advertised one"))?;
        let [a, b, c] = [0, 1, 2].map(|k| simbox.lengths()[k]);
        let [alpha, beta, gamma] = [0, 1, 2].map(|k| simbox.angles()[k]);
        let cos_alpha = alpha.to_radians().cos();
        let cos_beta = beta.to_radians().cos();
        let cos_gamma = gamma.to_radians().cos();

        write_marker(writer, meta.byte_order, meta.marker_size, 48)?;
        writer.write_all(&a.to_le_bytes())?;
        writer.write_all(&cos_gamma.to_le_bytes())?;
        writer.write_all(&b.to_le_bytes())?;
        writer.write_all(&cos_beta.to_le_bytes())?;
        writer.write_all(&cos_alpha.to_le_bytes())?;
        writer.write_all(&c.to_le_bytes())?;
        write_marker(writer, meta.byte_order, meta.marker_size, 48)?;
    }

    let natoms = meta.natoms as usize;
    let coord_record_len = (natoms as u64) * 4;

    let write_axis = |writer: &mut W, vals: &[f64]| -> std::io::Result<()> {
        write_marker(writer, meta.byte_order, meta.marker_size, coord_record_len)?;
        for &v in vals.iter().take(natoms) {
            writer.write_all(&(v as f32).to_le_bytes())?;
        }
        write_marker(writer, meta.byte_order, meta.marker_size, coord_record_len)?;
        Ok(())
    };

    let extract_axis = |key: &str| -> std::io::Result<Vec<f64>> {
        frame
            .column("atoms", key)
            .and_then(|c| c.get::<f64>())
            .map(|view| view.iter().copied().collect::<Vec<f64>>())
            .ok_or_else(|| invalid_data(format!("atoms.{} missing or not float", key)))
    };

    let xs = extract_axis("x")?;
    let ys = extract_axis("y")?;
    let zs = extract_axis("z")?;
    if xs.len() != natoms || ys.len() != natoms || zs.len() != natoms {
        return Err(invalid_data(format!(
            "atom count mismatch: x={}, y={}, z={}, expected {}",
            xs.len(),
            ys.len(),
            zs.len(),
            natoms
        )));
    }
    write_axis(writer, &xs)?;
    write_axis(writer, &ys)?;
    write_axis(writer, &zs)?;
    Ok(())
}

// ============================================================================
// Streaming
// ============================================================================

/// Magic prefix the WASM stream prepends when a decoder state is attached
/// (`setDecoderState`). Not a file format — never appears on disk.
const DCD_STATE_MAGIC: &[u8; 13] = b"MOLRS\x00DCDCTX\x00";

/// The first frame of DCD `bytes`, which are one of:
/// - a complete file starting at the CORD header;
/// - a frame body produced by [`DcdIndexBuilder`] (self-describing when
///   `NAMNF == 0`);
/// - that body prefixed with a decoder-state blob (needed for fixed-atom
///   later frames).
fn read_window(bytes: &[u8]) -> std::io::Result<Frame> {
    if let Some((header, rest)) = split_decoder_state(bytes)? {
        return parse_one_frame(
            &mut Cursor::new(rest),
            &header,
            frame_index_from_len(&header, rest),
        );
    }
    if looks_like_dcd_header(bytes) {
        return parse_header_and_first_frame(bytes);
    }
    parse_standalone_frame(bytes)
}

/// Decodes the frame bodies [`DcdIndexBuilder`] indexes, with the decoder
/// state it captured ([`FrameIndexBuilder::decoder_state`]: the header, and
/// for a fixed-atom file the frame-0 seed later frames need).
///
/// Holding the state apart from the frame bytes means no frame is copied to
/// prepend it.
pub struct DcdFrameDecoder {
    header: DcdHeader,
}

impl DcdFrameDecoder {
    /// A decoder for the file whose indexer captured `state`.
    ///
    /// # Errors
    ///
    /// `InvalidData` when `state` is not a decoder state.
    pub fn new(state: &[u8]) -> std::io::Result<Self> {
        Ok(Self {
            header: decode_decoder_state(state)?,
        })
    }

    /// Decode one frame body, as the indexer delimited it.
    pub fn decode(&self, frame: &[u8]) -> std::io::Result<Frame> {
        parse_frame_with_header(&self.header, frame)
    }
}

/// Decode `bytes` with an already-parsed header (fixed-atom later frames).
fn parse_frame_with_header(header: &DcdHeader, bytes: &[u8]) -> std::io::Result<Frame> {
    let n = frame_index_from_len(header, bytes);
    parse_one_frame(&mut Cursor::new(bytes), header, n)
}

fn frame_index_from_len(header: &DcdHeader, bytes: &[u8]) -> usize {
    if header.namnf > 0
        && header.frame_size_rest > 0
        && bytes.len() as u64 == header.frame_size_rest
    {
        1
    } else {
        0
    }
}

fn looks_like_dcd_header(bytes: &[u8]) -> bool {
    if bytes.len() < 12 {
        return false;
    }
    detect_endianness_and_marker(&mut Cursor::new(bytes)).is_ok()
}

fn parse_header_and_first_frame(bytes: &[u8]) -> std::io::Result<Frame> {
    let mut cursor = Cursor::new(bytes);
    let header = parse_header(&mut cursor)?;
    if header.nset == 0 {
        return Err(invalid_data("DCD header-only slice has no frames"));
    }
    cursor.seek(SeekFrom::Start(header.data_offset))?;
    parse_one_frame(&mut cursor, &header, 0)
}

fn parse_standalone_frame(bytes: &[u8]) -> std::io::Result<Frame> {
    let (order, marker, records) = detect_frame_records(bytes)?;
    if records.is_empty() {
        return Err(invalid_data("DCD frame slice has no Fortran records"));
    }

    let (box_payload, coord_recs) = split_box_and_coords(&records);
    if !(3..=4).contains(&coord_recs.len()) {
        return Err(invalid_data(format!(
            "DCD frame has {} coordinate records, expected 3 or 4",
            coord_recs.len()
        )));
    }
    let coord_len = coord_recs[0].len();
    if coord_len == 0 || !coord_len.is_multiple_of(4) {
        return Err(invalid_data(
            "DCD coordinate record length is not a multiple of 4",
        ));
    }
    if coord_recs.iter().any(|r| r.len() != coord_len) {
        return Err(invalid_data("DCD coordinate records have unequal lengths"));
    }

    let natoms = (coord_len / 4) as u32;
    let header = DcdHeader {
        byte_order: order,
        marker_size: marker,
        charmm_ver: 24,
        nset: 1,
        istart: 0,
        nsavc: 1,
        natoms,
        namnf: 0,
        has_box: box_payload.is_some(),
        has_4d: coord_recs.len() == 4,
        delta: 1.0,
        title: String::new(),
        free_atoms: None,
        fixed_seed: None,
        data_offset: 0,
        frame_size_first: bytes.len() as u64,
        frame_size_rest: bytes.len() as u64,
    };
    parse_one_frame(&mut Cursor::new(bytes), &header, 0)
}

fn split_box_and_coords(records: &[Vec<u8>]) -> (Option<&[u8]>, &[Vec<u8>]) {
    if records.len() >= 4 && records[0].len() == 48 {
        let rest_equal = records[1..].iter().all(|r| r.len() == records[1].len());
        if rest_equal && (records.len() == 4 || records.len() == 5) {
            return (Some(records[0].as_slice()), &records[1..]);
        }
    }
    (None, records)
}

fn detect_frame_records(bytes: &[u8]) -> std::io::Result<(ByteOrder, MarkerSize, Vec<Vec<u8>>)> {
    let mut best: Option<(ByteOrder, MarkerSize, Vec<Vec<u8>>)> = None;
    for &order in &[ByteOrder::Le, ByteOrder::Be] {
        for &marker in &[MarkerSize::Four, MarkerSize::Eight] {
            if let Some(recs) = try_read_all_records(bytes, order, marker) {
                let prefer_le4 = matches!((order, marker), (ByteOrder::Le, MarkerSize::Four));
                if best.is_none() || prefer_le4 {
                    best = Some((order, marker, recs));
                }
            }
        }
    }
    best.ok_or_else(|| invalid_data("not a standalone DCD frame: no consistent Fortran records"))
}

fn try_read_all_records(
    bytes: &[u8],
    order: ByteOrder,
    marker: MarkerSize,
) -> Option<Vec<Vec<u8>>> {
    let mut pos = 0usize;
    let mut recs = Vec::new();
    while pos < bytes.len() {
        match try_read_record_at(bytes, pos, order, marker) {
            Some((payload, next)) => {
                recs.push(payload);
                pos = next;
            }
            None => return None,
        }
    }
    if recs.is_empty() { None } else { Some(recs) }
}

/// Read one Fortran record without allocating more than the remaining slice.
fn try_read_record_at(
    bytes: &[u8],
    pos: usize,
    order: ByteOrder,
    marker: MarkerSize,
) -> Option<(Vec<u8>, usize)> {
    let m = marker.bytes() as usize;
    if pos.checked_add(2 * m)? > bytes.len() {
        return None;
    }
    let leading = match marker {
        MarkerSize::Four => {
            let raw: [u8; 4] = bytes[pos..pos + 4].try_into().ok()?;
            read_u32(&raw, order) as u64
        }
        MarkerSize::Eight => {
            let raw: [u8; 8] = bytes[pos..pos + 8].try_into().ok()?;
            match order {
                ByteOrder::Le => u64::from_le_bytes(raw),
                ByteOrder::Be => u64::from_be_bytes(raw),
            }
        }
    };
    let payload_len = usize::try_from(leading).ok()?;
    let payload_start = pos + m;
    let payload_end = payload_start.checked_add(payload_len)?;
    let trailing_end = payload_end.checked_add(m)?;
    if trailing_end > bytes.len() {
        return None;
    }
    let trailing = match marker {
        MarkerSize::Four => {
            let raw: [u8; 4] = bytes[payload_end..payload_end + 4].try_into().ok()?;
            read_u32(&raw, order) as u64
        }
        MarkerSize::Eight => {
            let raw: [u8; 8] = bytes[payload_end..payload_end + 8].try_into().ok()?;
            match order {
                ByteOrder::Le => u64::from_le_bytes(raw),
                ByteOrder::Be => u64::from_be_bytes(raw),
            }
        }
    };
    if leading != trailing {
        return None;
    }
    Some((bytes[payload_start..payload_end].to_vec(), trailing_end))
}

fn split_decoder_state(bytes: &[u8]) -> std::io::Result<Option<(DcdHeader, &[u8])>> {
    if bytes.len() < DCD_STATE_MAGIC.len() + 4 || !bytes.starts_with(DCD_STATE_MAGIC) {
        return Ok(None);
    }
    let len_off = DCD_STATE_MAGIC.len();
    let state_len = u32::from_le_bytes(bytes[len_off..len_off + 4].try_into().unwrap()) as usize;
    if state_len > MAX_DCD_RECORD_BYTES as usize {
        return Err(invalid_data("DCD decoder state larger than the record cap"));
    }
    let state_start = len_off + 4;
    let state_end = state_start
        .checked_add(state_len)
        .ok_or_else(|| invalid_data("DCD decoder state length overflow"))?;
    if state_end > bytes.len() {
        return Err(invalid_data("DCD decoder state truncated"));
    }
    let header = decode_decoder_state(&bytes[state_start..state_end])?;
    Ok(Some((header, &bytes[state_end..])))
}

fn encode_decoder_state(header: &DcdHeader) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(1u8);
    out.push(match header.byte_order {
        ByteOrder::Le => 0,
        ByteOrder::Be => 1,
    });
    out.push(match header.marker_size {
        MarkerSize::Four => 4,
        MarkerSize::Eight => 8,
    });
    out.extend_from_slice(&header.charmm_ver.to_le_bytes());
    out.extend_from_slice(&header.nset.to_le_bytes());
    out.extend_from_slice(&header.istart.to_le_bytes());
    out.extend_from_slice(&header.nsavc.to_le_bytes());
    out.extend_from_slice(&header.natoms.to_le_bytes());
    out.extend_from_slice(&header.namnf.to_le_bytes());
    out.push(u8::from(header.has_box));
    out.push(u8::from(header.has_4d));
    out.extend_from_slice(&header.delta.to_le_bytes());
    let title = header.title.as_bytes();
    out.extend_from_slice(&(title.len() as u32).to_le_bytes());
    out.extend_from_slice(title);
    match &header.free_atoms {
        Some(indices) => {
            out.extend_from_slice(&(indices.len() as u32).to_le_bytes());
            for idx in indices {
                out.extend_from_slice(&idx.to_le_bytes());
            }
        }
        None => out.extend_from_slice(&0u32.to_le_bytes()),
    }
    match &header.fixed_seed {
        Some((xs, ys, zs)) => {
            out.push(1);
            let n = header.natoms as usize;
            for axis in [xs, ys, zs] {
                for i in 0..n {
                    let v = axis.get(i).copied().unwrap_or(0.0);
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        None => out.push(0),
    }
    out.extend_from_slice(&header.data_offset.to_le_bytes());
    out.extend_from_slice(&header.frame_size_first.to_le_bytes());
    out.extend_from_slice(&header.frame_size_rest.to_le_bytes());
    out
}

fn decode_decoder_state(bytes: &[u8]) -> std::io::Result<DcdHeader> {
    let mut cur = Cursor::new(bytes);
    let mut ver = [0u8; 1];
    cur.read_exact(&mut ver)?;
    if ver[0] != 1 {
        return Err(invalid_data(format!(
            "unsupported DCD decoder state version {}",
            ver[0]
        )));
    }
    let mut flags = [0u8; 2];
    cur.read_exact(&mut flags)?;
    let byte_order = match flags[0] {
        0 => ByteOrder::Le,
        1 => ByteOrder::Be,
        _ => return Err(invalid_data("bad DCD decoder state byte order")),
    };
    let marker_size = match flags[1] {
        4 => MarkerSize::Four,
        8 => MarkerSize::Eight,
        _ => return Err(invalid_data("bad DCD decoder state marker size")),
    };
    let mut i32b = [0u8; 4];
    let mut u32b = [0u8; 4];
    cur.read_exact(&mut i32b)?;
    let charmm_ver = i32::from_le_bytes(i32b);
    cur.read_exact(&mut u32b)?;
    let nset = u32::from_le_bytes(u32b);
    cur.read_exact(&mut i32b)?;
    let istart = i32::from_le_bytes(i32b);
    cur.read_exact(&mut i32b)?;
    let nsavc = i32::from_le_bytes(i32b);
    cur.read_exact(&mut u32b)?;
    let natoms = u32::from_le_bytes(u32b);
    cur.read_exact(&mut u32b)?;
    let namnf = u32::from_le_bytes(u32b);
    if namnf > natoms {
        return Err(invalid_data(format!(
            "DCD decoder state namnf {namnf} > natoms {natoms}"
        )));
    }
    let mut hb = [0u8; 2];
    cur.read_exact(&mut hb)?;
    let mut dlt = [0u8; 8];
    cur.read_exact(&mut dlt)?;
    cur.read_exact(&mut u32b)?;
    let title_len = u32::from_le_bytes(u32b) as usize;
    let remaining = bytes.len().saturating_sub(cur.position() as usize);
    if title_len > remaining || title_len > 1_048_576 {
        return Err(invalid_data(format!(
            "DCD decoder state title length {title_len} is not credible"
        )));
    }
    let mut title = vec![0u8; title_len];
    cur.read_exact(&mut title)?;
    cur.read_exact(&mut u32b)?;
    let nfree = u32::from_le_bytes(u32b) as usize;
    if nfree > natoms as usize {
        return Err(invalid_data(format!(
            "DCD decoder state nfree {nfree} > natoms {natoms}"
        )));
    }
    let remaining = bytes.len().saturating_sub(cur.position() as usize);
    if nfree.saturating_mul(4) > remaining {
        return Err(invalid_data("DCD decoder state free-atom list truncated"));
    }
    let free_atoms = if nfree == 0 {
        None
    } else {
        let mut v = Vec::with_capacity(nfree);
        for _ in 0..nfree {
            cur.read_exact(&mut i32b)?;
            v.push(i32::from_le_bytes(i32b));
        }
        Some(v)
    };
    let mut has_seed = [0u8; 1];
    cur.read_exact(&mut has_seed)?;
    let fixed_seed = if has_seed[0] == 1 {
        let n = natoms as usize;
        let need = n.saturating_mul(12);
        let remaining = bytes.len().saturating_sub(cur.position() as usize);
        if need > remaining {
            return Err(invalid_data("DCD decoder state fixed-atom seed truncated"));
        }
        let mut read_axis = || -> std::io::Result<Vec<f32>> {
            let mut axis = vec![0.0f32; n];
            for slot in &mut axis {
                let mut b = [0u8; 4];
                cur.read_exact(&mut b)?;
                *slot = f32::from_le_bytes(b);
            }
            Ok(axis)
        };
        Some((read_axis()?, read_axis()?, read_axis()?))
    } else {
        None
    };
    let mut u64b = [0u8; 8];
    cur.read_exact(&mut u64b)?;
    let data_offset = u64::from_le_bytes(u64b);
    cur.read_exact(&mut u64b)?;
    let frame_size_first = u64::from_le_bytes(u64b);
    cur.read_exact(&mut u64b)?;
    let frame_size_rest = u64::from_le_bytes(u64b);

    Ok(DcdHeader {
        byte_order,
        marker_size,
        charmm_ver,
        nset,
        istart,
        nsavc,
        natoms,
        namnf,
        has_box: hb[0] != 0,
        has_4d: hb[1] != 0,
        delta: f64::from_le_bytes(dlt),
        title: String::from_utf8_lossy(&title).into_owned(),
        free_atoms,
        fixed_seed,
        data_offset,
        frame_size_first,
        frame_size_rest,
    })
}

/// Streaming frame indexer for DCD files.
///
/// After the header is known, frame sizes are constant: further `feed`
/// calls only advance the byte counter and emit arithmetic entries. The
/// whole file is never buffered.
pub struct DcdIndexBuilder {
    buf: Vec<u8>,
    buf_start: u64,
    bytes_seen: u64,
    finished: bool,
    total_hint: Option<u64>,
    partial: Option<HeaderPartial>,
    header: Option<DcdHeader>,
    next_frame: usize,
    pending: Vec<FrameOffset>,
    error: Option<std::io::Error>,
}

impl Default for DcdIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DcdIndexBuilder {
    /// Create an empty indexer.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            buf_start: 0,
            bytes_seen: 0,
            finished: false,
            total_hint: None,
            partial: None,
            header: None,
            next_frame: 0,
            pending: Vec::new(),
            error: None,
        }
    }

    fn fail(&mut self, err: std::io::Error) {
        if self.error.is_none() {
            self.error = Some(err);
        }
    }

    fn append(&mut self, chunk: &[u8], global_offset: u64) {
        if chunk.is_empty() {
            self.bytes_seen = self.bytes_seen.max(global_offset);
            return;
        }
        if self.header.is_some() && self.seed_ready() {
            // Layout is arithmetic from here — keep bytes_seen monotonic
            // even if a host re-feeds an overlapping window.
            self.bytes_seen = self
                .bytes_seen
                .max(global_offset.saturating_add(chunk.len() as u64));
            return;
        }
        if self.buf.is_empty() {
            self.buf_start = global_offset;
            self.buf.extend_from_slice(chunk);
        } else {
            let expected = self.buf_start + self.buf.len() as u64;
            if global_offset == expected {
                self.buf.extend_from_slice(chunk);
            } else if global_offset < expected {
                let skip = (expected - global_offset) as usize;
                if skip < chunk.len() {
                    self.buf.extend_from_slice(&chunk[skip..]);
                }
            } else {
                self.buf.clear();
                self.buf_start = global_offset;
                self.buf.extend_from_slice(chunk);
            }
        }
        self.bytes_seen = global_offset.saturating_add(chunk.len() as u64);
    }

    fn seed_ready(&self) -> bool {
        match &self.header {
            None => false,
            Some(h) => h.namnf == 0 || h.fixed_seed.is_some(),
        }
    }

    fn try_progress(&mut self) {
        if self.error.is_some() {
            return;
        }
        if self.header.is_none() {
            if self.partial.is_none() {
                match try_parse_header_records(&self.buf) {
                    Ok(Some(partial)) => {
                        let drop = partial.data_offset.saturating_sub(self.buf_start) as usize;
                        if drop > 0 && drop <= self.buf.len() {
                            self.buf.drain(..drop);
                            self.buf_start += drop as u64;
                        }
                        self.partial = Some(partial);
                    }
                    Ok(None) => return,
                    Err(err) => {
                        self.fail(err);
                        return;
                    }
                }
            }
            if let Some(partial) = self.partial.clone() {
                match self.resolve_layout(&partial) {
                    Ok(Some(layout)) => {
                        self.header = Some(partial.into_header(layout));
                        self.partial = None;
                    }
                    Ok(None) => return,
                    Err(err) => {
                        self.fail(err);
                        return;
                    }
                }
            }
        }
        self.capture_fixed_seed();
        self.emit_complete_frames();
        if self.seed_ready() {
            self.buf.clear();
        }
    }

    fn resolve_layout(&self, partial: &HeaderPartial) -> std::io::Result<Option<FrameLayout>> {
        let m = partial.marker_size.bytes() as usize;
        if self.buf.len() < m {
            return Ok(None);
        }
        let mut cursor = Cursor::new(&self.buf);
        let first_marker = read_marker(&mut cursor, partial.byte_order, partial.marker_size)?;

        let known_total = self.total_hint.or(if self.finished {
            Some(self.bytes_seen)
        } else {
            None
        });
        if let Some(total) = known_total {
            let trailing = total.saturating_sub(partial.data_offset);
            return resolve_frame_layout(partial, first_marker, trailing).map(Some);
        }
        // A CHARMM header states the layout, so no total is needed.
        if let Some((has_box, has_4d)) = partial.charmm_flags {
            let (frame_size_first, frame_size_rest) =
                layout_from_flags(partial, first_marker, has_box, has_4d)?;
            return Ok(Some(FrameLayout {
                has_box,
                has_4d,
                frame_size_first,
                frame_size_rest,
                nset: 0,
            }));
        }

        // Incremental: a boxed file (first marker 48, natoms ≠ 12) can
        // resolve has_4d by peeking the record after three coordinate axes.
        let coord_marker_full = u64::from(partial.natoms) * 4;
        if first_marker == 48 && partial.natoms != 12 {
            let rec = 2 * partial.marker_size.bytes() + coord_marker_full;
            let box_rec = 2 * partial.marker_size.bytes() + 48;
            let three = box_rec + 3 * rec;
            if self.bytes_seen < partial.data_offset + three + partial.marker_size.bytes() {
                return Ok(None);
            }
            let peek_at = three as usize;
            if self.buf.len() < peek_at + m {
                return Ok(None);
            }
            let mut peek = Cursor::new(&self.buf[peek_at..]);
            let next = read_marker(&mut peek, partial.byte_order, partial.marker_size)?;
            let has_4d = next == coord_marker_full;
            let n_recs = if has_4d { 4 } else { 3 };
            let f_first = box_rec + rec * n_recs;
            let rec_eff =
                2 * partial.marker_size.bytes() + u64::from(partial.natoms - partial.namnf) * 4;
            let f_rest = box_rec + rec_eff * n_recs;
            return Ok(Some(FrameLayout {
                has_box: true,
                has_4d,
                frame_size_first: f_first,
                frame_size_rest: f_rest,
                nset: 0,
            }));
        }
        Ok(None)
    }

    fn capture_fixed_seed(&mut self) {
        let Some(header) = self.header.as_mut() else {
            return;
        };
        if header.namnf == 0 || header.fixed_seed.is_some() {
            return;
        }
        let need = header.frame_size_first as usize;
        if need == 0 || self.buf.len() < need {
            return;
        }
        let mut cursor = Cursor::new(&self.buf[..need]);
        if let Ok((xs, ys, zs, _)) = read_coord_payload(&mut cursor, header, 0) {
            header.fixed_seed = Some((xs, ys, zs));
        }
    }

    fn emit_complete_frames(&mut self) {
        let Some(header) = self.header.as_ref() else {
            return;
        };
        if header.frame_size_first == 0 {
            return;
        }
        loop {
            let start = header.frame_offset(self.next_frame);
            let size = if self.next_frame == 0 {
                header.frame_size_first
            } else {
                header.frame_size_rest
            };
            if size == 0 || size > u64::from(u32::MAX) {
                break;
            }
            let end = start.saturating_add(size);
            if self.bytes_seen < end {
                break;
            }
            self.pending.push(FrameOffset {
                byte_offset: start,
                byte_len: size as u32,
            });
            self.next_frame += 1;
        }
    }
}

impl FrameIndexBuilder for DcdIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        if self.finished {
            panic!("DcdIndexBuilder::feed called after finish");
        }
        self.append(chunk, global_offset);
        self.try_progress();
    }

    fn drain(&mut self) -> Vec<FrameOffset> {
        std::mem::take(&mut self.pending)
    }

    fn finish(mut self: Box<Self>) -> std::io::Result<Vec<FrameOffset>> {
        self.finished = true;
        self.try_progress();
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        if self.header.is_none() && !self.buf.is_empty() {
            return Err(invalid_data("truncated DCD header"));
        }
        Ok(std::mem::take(&mut self.pending))
    }

    fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }

    fn hint_total_bytes(&mut self, total: u64) {
        self.total_hint = Some(total);
        self.try_progress();
    }

    fn decoder_state(&self) -> Option<Vec<u8>> {
        // Fixed-atom later frames need the frame-0 seed. Handing the
        // header out before that seed is captured would decode as if
        // every atom were free.
        if !self.seed_ready() {
            return None;
        }
        self.header.as_ref().map(encode_decoder_state)
    }
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Read the first frame of a DCD file. Bytes may also be one frame body
/// as [`DcdIndexBuilder`] delimits it (self-describing when `NAMNF == 0`), or
/// that body prefixed with the indexer's decoder state.
pub fn read_dcd<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Frame> {
    let src = src.into();
    if let Source::Bytes(bytes) = src {
        return read_window(bytes);
    }
    DcdReader::new(src.reader()?)
        .read_frame(0)?
        .ok_or_else(|| invalid_data(format!("{}: DCD file holds no frame", src.name())))
}

/// Write one frame as a complete one-frame DCD file. 4D-dynamics frames (a
/// `w` column on `atoms`) are refused with `ErrorKind::Unsupported`.
pub fn write_dcd<'a>(sink: impl Into<Sink<'a>>, frame: &impl FrameAccess) -> std::io::Result<()> {
    sink.into().put_with(|w| {
        let mut dcd = DcdWriter::new(w);
        write_dcd_frame(&mut dcd, frame)?;
        dcd.writer.flush()
    })
}

impl DcdReader<Box<dyn ReadSeek>> {
    /// Open a DCD file, or the bytes of one, for random access by frame.
    pub fn open<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Self> {
        Ok(Self::new(src.into().owned_reader()?))
    }
}

impl<'a> DcdWriter<SinkWriter<'a>> {
    /// Create a DCD file, or fill a buffer, frame by frame. All frames must
    /// have the first frame's atom count and box presence.
    pub fn create(sink: impl Into<Sink<'a>>) -> std::io::Result<Self> {
        Ok(Self::new(sink.into().writer()?))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use std::io::Cursor;

    #[test]
    fn test_detect_endianness_and_marker_le4() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&84u32.to_le_bytes());
        buf.extend_from_slice(b"CORD");
        buf.extend_from_slice(&[0u8; 4]);
        let mut cur = Cursor::new(buf);
        assert_eq!(
            detect_endianness_and_marker(&mut cur).unwrap(),
            (ByteOrder::Le, MarkerSize::Four)
        );
    }

    #[test]
    fn test_detect_endianness_and_marker_be4() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&84u32.to_be_bytes());
        buf.extend_from_slice(b"CORD");
        buf.extend_from_slice(&[0u8; 4]);
        let mut cur = Cursor::new(buf);
        assert_eq!(
            detect_endianness_and_marker(&mut cur).unwrap(),
            (ByteOrder::Be, MarkerSize::Four)
        );
    }

    #[test]
    fn test_detect_endianness_and_marker_le8() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&84u64.to_le_bytes());
        buf.extend_from_slice(b"CORD");
        let mut cur = Cursor::new(buf);
        assert_eq!(
            detect_endianness_and_marker(&mut cur).unwrap(),
            (ByteOrder::Le, MarkerSize::Eight)
        );
    }

    #[test]
    fn test_detect_endianness_and_marker_be8() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&84u64.to_be_bytes());
        buf.extend_from_slice(b"CORD");
        let mut cur = Cursor::new(buf);
        assert_eq!(
            detect_endianness_and_marker(&mut cur).unwrap(),
            (ByteOrder::Be, MarkerSize::Eight)
        );
    }

    #[test]
    fn test_detect_endianness_and_marker_garbage() {
        let mut cur = Cursor::new(vec![0u8; 12]);
        assert!(detect_endianness_and_marker(&mut cur).is_err());
    }

    #[test]
    fn test_box_cosine_vs_degree_heuristic_agree() {
        // Cubic box: A=B=C=10, alpha=beta=gamma=90°.
        let mut deg_payload = Vec::with_capacity(48);
        for v in [10.0f64, 90.0, 10.0, 90.0, 90.0, 10.0] {
            deg_payload.extend_from_slice(&v.to_le_bytes());
        }
        let mut cos_payload = Vec::with_capacity(48);
        for v in [10.0f64, 0.0, 10.0, 0.0, 0.0, 10.0] {
            cos_payload.extend_from_slice(&v.to_le_bytes());
        }
        let bx_deg = parse_box_payload(&deg_payload, ByteOrder::Le).unwrap();
        let bx_cos = parse_box_payload(&cos_payload, ByteOrder::Le).unwrap();
        let lengths_deg = bx_deg.lengths();
        let lengths_cos = bx_cos.lengths();
        for i in 0..3 {
            assert!((lengths_deg[i] - lengths_cos[i]).abs() < 1e-9);
            assert!((lengths_deg[i] - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_fortran_record_marker_mismatch() {
        let mut buf = vec![];
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&7u32.to_le_bytes()); // mismatch on purpose
        let mut cur = Cursor::new(buf);
        let err = read_record(&mut cur, ByteOrder::Le, MarkerSize::Four).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_triclinic_box_record_round_trips_its_lengths_and_angles() {
        let payload: Vec<u8> = [
            10.0f64,
            95.0f64.to_radians().cos(),
            12.0,
            80.0f64.to_radians().cos(),
            70.0f64.to_radians().cos(),
            15.0,
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
        let simbox = parse_box_payload(&payload, ByteOrder::Le).unwrap();
        let (l, a) = (simbox.lengths(), simbox.angles());
        for (got, want) in l.iter().zip([10.0, 12.0, 15.0]) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        for (got, want) in a.iter().zip([70.0, 80.0, 95.0]) {
            assert!((got - want).abs() < 1e-7, "{got} vs {want}");
        }
    }

    fn two_atom_frame(x0: f64, with_box: bool) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert("id", Array1::from(vec![1u64, 2]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("x", Array1::from(vec![x0, x0 + 1.0]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("y", Array1::from(vec![0.0, 0.0]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("z", Array1::from(vec![0.0, 0.0]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        if with_box {
            let h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
            let origin = [0.0, 0.0, 0.0];
            frame.simbox = Some(SimBox::new(Cell::Matrix(h), origin, [true; 3]).unwrap());
        }
        frame
    }

    /// `n` atoms on the x axis from `x0`, one Å apart.
    fn n_atom_frame(n: usize, x0: f64, with_box: bool) -> Frame {
        let mut atoms = Block::new();
        let ids: Vec<u64> = (1..=n as u64).collect();
        let xs: Vec<f64> = (0..n).map(|i| x0 + i as f64).collect();
        atoms
            .insert("id", Array1::from(ids).into_dyn(), None)
            .unwrap();
        atoms
            .insert("x", Array1::from(xs).into_dyn(), None)
            .unwrap();
        atoms
            .insert("y", Array1::from(vec![0.0; n]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("z", Array1::from(vec![0.0; n]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        if with_box {
            let h = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]];
            frame.simbox = Some(SimBox::new(Cell::Matrix(h), [0.0; 3], [true; 3]).unwrap());
        }
        frame
    }

    fn write_dcd_mem(frames: &[Frame]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = DcdWriter::new(&mut cursor);
            for frame in frames {
                FrameWriter::write(&mut writer, frame).expect("write dcd frame");
            }
        }
        cursor.into_inner()
    }

    fn dcd_index_chunked(bytes: &[u8], chunk: usize) -> Vec<FrameOffset> {
        let mut builder = Box::new(DcdIndexBuilder::new());
        builder.hint_total_bytes(bytes.len() as u64);
        let mut offset = 0u64;
        let mut out = Vec::new();
        for piece in bytes.chunks(chunk.max(1)) {
            builder.feed(piece, offset);
            offset += piece.len() as u64;
            out.extend(builder.drain());
        }
        out.extend(builder.finish().expect("finish dcd index"));
        out
    }

    /// The CHARMM header words: DELTA at `icntrl[9]`, the crystal flag at
    /// `icntrl[10]`, the 4D flag at `icntrl[11]`; the reader's `delta` is the
    /// writer's.
    #[test]
    fn the_header_puts_delta_and_the_crystal_flag_in_the_charmm_words() {
        let bytes = write_dcd_mem(&[two_atom_frame(0.0, true)]);
        // Record 1: a 4-byte marker, "CORD", then icntrl[i] at 8 + 4·i.
        let word = |i: usize| -> [u8; 4] { bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap() };
        assert_eq!(f32::from_le_bytes(word(9)), 1.0, "DELTA");
        assert_eq!(i32::from_le_bytes(word(10)), 1, "crystal flag");
        assert_eq!(i32::from_le_bytes(word(11)), 0, "4D flag");
        let mut reader = DcdReader::new(Cursor::new(bytes));
        let frame = reader.read_frame(0).unwrap().unwrap();
        assert_eq!(frame.meta.get("delta").and_then(|v| v.as_str()), Some("1"));
    }

    /// Twelve atoms make a box record the size of a coordinate record, so
    /// the byte count alone cannot tell three boxed frames from four box-less
    /// ones: the CHARMM header flags decide.
    #[test]
    fn twelve_boxed_atoms_over_three_frames_read_as_three_boxed_frames() {
        let frames: Vec<Frame> = (0..3).map(|i| n_atom_frame(12, i as f64, true)).collect();
        let bytes = write_dcd_mem(&frames);
        let mut reader = DcdReader::new(Cursor::new(bytes));
        assert_eq!(reader.len().unwrap(), 3);
        for i in 0..3 {
            let frame = reader.read_frame(i).unwrap().unwrap();
            assert!(frame.simbox.is_some(), "frame {i} lost its box");
            let x = frame
                .get("atoms")
                .unwrap()
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap();
            assert!((x[0] - i as f64).abs() < 1e-5, "frame {i}: {x:?}");
        }
    }

    #[test]
    fn dcd_streaming_boxed_single_shot_matches_chunks() {
        let frames = [two_atom_frame(0.0, true), two_atom_frame(0.5, true)];
        let bytes = write_dcd_mem(&frames);
        let one_shot = dcd_index_chunked(&bytes, bytes.len());
        assert_eq!(one_shot.len(), 2, "expected 2 DCD frames");
        for cs in [1usize, 7, 17, 64, 256] {
            assert_eq!(
                dcd_index_chunked(&bytes, cs),
                one_shot,
                "chunk size {cs} produced a different index"
            );
        }
        for (i, entry) in one_shot.iter().enumerate() {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            let parsed = read_dcd(&bytes[lo..hi]).expect("parse dcd frame");
            let x = parsed
                .get("atoms")
                .unwrap()
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap();
            assert!(
                (x[0]
                    - frames[i]
                        .get("atoms")
                        .unwrap()
                        .get("x")
                        .and_then(|c| c.get::<f64>())
                        .unwrap()[0])
                    .abs()
                    < 1e-5
            );
        }
    }

    #[test]
    fn dcd_streaming_nopbc_indexes_at_finish() {
        let frames = [two_atom_frame(1.0, false), two_atom_frame(2.0, false)];
        let bytes = write_dcd_mem(&frames);
        let entries = dcd_index_chunked(&bytes, 13);
        assert_eq!(entries.len(), 2);
        let f0 = read_dcd(
            &bytes[entries[0].byte_offset as usize
                ..entries[0].byte_offset as usize + entries[0].byte_len as usize],
        )
        .expect("parse nopbc frame 0");
        assert!(f0.simbox.is_none());
        assert_eq!(f0.get("atoms").unwrap().n_rows().unwrap(), 2);
    }

    #[test]
    fn dcd_truncated_last_frame_keeps_complete_frames() {
        let frames = [
            two_atom_frame(0.0, true),
            two_atom_frame(0.5, true),
            two_atom_frame(1.0, true),
        ];
        let full = write_dcd_mem(&frames);
        let full_entries = dcd_index_chunked(&full, full.len());
        assert_eq!(
            full_entries.len(),
            3,
            "fixture must write 3 complete frames"
        );
        let last = &full_entries[2];
        let mut bytes = full;
        bytes.truncate((last.byte_offset + u64::from(last.byte_len) / 2) as usize);
        let entries = dcd_index_chunked(&bytes, bytes.len());
        assert_eq!(
            entries.len(),
            2,
            "truncated last frame must not drop the complete ones"
        );
        let mut reader = DcdReader::new(Cursor::new(bytes));
        assert_eq!(reader.len().expect("len"), 2);
        assert!(reader.read_frame(0).expect("frame 0").is_some());
        assert!(reader.read_frame(1).expect("frame 1").is_some());
        assert!(reader.read_frame(2).expect("frame 2").is_none());
    }

    #[test]
    fn dcd_streaming_boxed_emits_first_frame_before_eof() {
        let frames = [two_atom_frame(0.0, true), two_atom_frame(1.0, true)];
        let bytes = write_dcd_mem(&frames);
        let mut builder = DcdIndexBuilder::new();
        // No total hint: boxed layout resolves after the first frame + next marker.
        builder.feed(&bytes, 0);
        let early = builder.drain();
        assert!(
            !early.is_empty(),
            "boxed DCD should emit frame 0 before finish"
        );
        assert!(early[0].byte_offset > 0, "frame 0 starts after the header");
        let rest = Box::new(builder).finish().expect("finish");
        assert_eq!(early.len() + rest.len(), 2);
    }

    #[test]
    fn dcd_decoder_state_round_trips_header() {
        let bytes = write_dcd_mem(&[two_atom_frame(0.0, true)]);
        let mut builder = Box::new(DcdIndexBuilder::new());
        builder.hint_total_bytes(bytes.len() as u64);
        builder.feed(&bytes, 0);
        let state = builder.decoder_state().expect("decoder state");
        let entries = builder.finish().expect("finish");
        assert_eq!(entries.len(), 1);
        let mut wrapped = Vec::from(DCD_STATE_MAGIC.as_slice());
        wrapped.extend_from_slice(&(state.len() as u32).to_le_bytes());
        wrapped.extend_from_slice(&state);
        let lo = entries[0].byte_offset as usize;
        let hi = lo + entries[0].byte_len as usize;
        wrapped.extend_from_slice(&bytes[lo..hi]);
        let parsed = read_dcd(&wrapped).expect("parse with decoder state");
        assert_eq!(parsed.get("atoms").unwrap().n_rows().unwrap(), 2);

        let no_copy = DcdFrameDecoder::new(&state)
            .and_then(|d| d.decode(&bytes[lo..hi]))
            .expect("parse with decoder state, no wrap");
        assert_eq!(no_copy.get("atoms").unwrap().n_rows().unwrap(), 2);
    }

    #[test]
    fn dcd_decoder_state_rejects_huge_title() {
        let mut bad = vec![1u8, 0, 4]; // ver, le, marker 4
        bad.extend_from_slice(&24i32.to_le_bytes()); // charmm
        bad.extend_from_slice(&1u32.to_le_bytes()); // nset
        bad.extend_from_slice(&0i32.to_le_bytes()); // istart
        bad.extend_from_slice(&1i32.to_le_bytes()); // nsavc
        bad.extend_from_slice(&2u32.to_le_bytes()); // natoms
        bad.extend_from_slice(&0u32.to_le_bytes()); // namnf
        bad.extend_from_slice(&[0u8, 0]); // has_box, has_4d
        bad.extend_from_slice(&1.0f64.to_le_bytes());
        bad.extend_from_slice(&2_000_000u32.to_le_bytes()); // title_len
        assert!(decode_decoder_state(&bad).is_err());
    }

    #[test]
    fn dcd_read_record_rejects_huge_marker() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(MAX_DCD_RECORD_BYTES as u32 + 1).to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        let mut cur = Cursor::new(buf);
        let err = read_record(&mut cur, ByteOrder::Le, MarkerSize::Four).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("exceeds cap"));
    }
}
