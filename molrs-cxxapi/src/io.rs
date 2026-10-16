//! File I/O for the C++ engine — the CXX face of `molrs::io`: XYZ / ExtXYZ
//! frames, `*.mrec` frame records, frames of a `*.mrec` trajectory and the
//! streaming `*.mrec` trajectory writer.
//!
//! One frame currency crosses in both directions: every writer takes the
//! caller's [`FrameRef`] (never modified — a writer that needs a derived
//! column works on a copy) and every reader returns a fresh one. Species is
//! the canonical `atomic_number` Uint column and conversion is always
//! explicit: XYZ spells symbols, so [`write_xyz`] calls
//! [`Element::fill_symbols`] on its copy and [`read_xyz`] calls
//! [`Element::fill_atomic_numbers`]; `*.mrec` is molrs' native format and
//! stores `atomic_number` as is.

use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::path::Path;

use molrs::core::Element;
use molrs::core::Frame;
use molrs::core::keys;
use molrs::io::mrec::{MolRec, MrecFile, MrecReader, MrecWriter, SequenceSchema};
use molrs::io::writer::{FrameWriter, Writer};
use molrs::io::xyz::XyzWriter;

use crate::frame::FrameRef;

/// Write one frame (its atoms columns, box and exact-dtype metadata) to an
/// XYZ / ExtXYZ file through molrs `XyzWriter`; `append` adds it after the
/// frames already there, otherwise the file is truncated.
///
/// XYZ spells species as symbols, so the `element` column the writer needs is
/// filled explicitly from `atoms.atomic_number` by [`Element::fill_symbols`]
/// (Z = 0 is written as the wildcard `*`) — on a copy: the caller's frame is
/// not touched.
///
/// # Errors
/// A frame without an `atoms.atomic_number` Uint column, a number outside
/// `0..=118` (the row is named), or the I/O / vocabulary error of the writer.
pub(crate) fn write_xyz(path: &str, fref: &FrameRef, append: bool) -> Result<(), String> {
    let what = "write_xyz";
    let mut frame = fref.0.clone_frame().map_err(|e| format!("{what}: {e}"))?;
    Element::fill_symbols(&mut frame).map_err(|e| format!("{what}: {e}"))?;
    let file = if append {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("{what}: open {path}: {e}"))?
    } else {
        File::create(path).map_err(|e| format!("{what}: create {path}: {e}"))?
    };
    XyzWriter::new(BufWriter::new(file))
        .write(&frame)
        .map_err(|e| format!("{what}: {e}"))
}

/// Refuse a frame whose `atoms.atomic_number` holds a number that is not an
/// atomic number of the schema (`0..=118`, 0 = wildcard `*`), naming the row.
///
/// The native `*.mrec` doors store the column as is, and the molrs frame
/// vocabulary binds its dtype but not its range — so they check it here, as
/// [`Element::fill_symbols`] does for the symbol formats. A frame without the
/// column passes: what it does not carry is not wrong.
fn require_atomic_numbers(frame: &Frame) -> Result<(), String> {
    let Some(numbers) = frame
        .get("atoms")
        .and_then(|atoms| atoms.get(keys::ATOMIC_NUMBER))
        .and_then(|column| column.get::<u64>())
    else {
        return Ok(());
    };
    let is_atomic_number =
        |z: u64| u8::try_from(z).is_ok_and(|z| z == 0 || Element::by_number(z).is_some());
    match numbers
        .iter()
        .enumerate()
        .find(|&(_, &z)| !is_atomic_number(z))
    {
        Some((row, z)) => Err(format!(
            "atoms row {row}: {z} is not an atomic number (0..=118)"
        )),
        None => Ok(()),
    }
}

/// Write one frame as a `*.mrec` record whose `frame` section is that frame —
/// molrs `io::write_mrec` with `record.frame` set. Every block and column, the
/// declared column precisions, the box and the metadata are stored; a store
/// already at `path` is replaced.
///
/// `*.mrec` is molrs' native format: species is stored as the canonical
/// `atomic_number` column itself (no `element` symbols are derived), so a
/// round trip hands back exactly the field every molrs consumer reads.
///
/// # Errors
/// A number outside `0..=118` in `atoms.atomic_number` (the row is named), or
/// the molrs record error (a column outside the vocabulary, an unwritable
/// path, a retired `.zarr` suffix).
pub(crate) fn write_mrec_frame(path: &str, fref: &FrameRef) -> Result<(), String> {
    let what = "write_mrec_frame";
    let frame = fref.0.clone_frame().map_err(|e| format!("{what}: {e}"))?;
    require_atomic_numbers(&frame).map_err(|e| format!("{what}: {e}"))?;
    let record = MolRec {
        frame: Some(frame),
        ..MolRec::default()
    };
    molrs::io::write_mrec(Path::new(path), &record).map_err(|e| format!("{what}: {e}"))
}

/// Hand a frame read from disk to C++ as a fresh standalone [`FrameRef`].
fn frame_ref_holding(frame: Frame, what: &str) -> Result<Box<FrameRef>, String> {
    let inner = molrs_ffi::FrameRef::default();
    inner
        .with_mut(|f| *f = frame)
        .map_err(|e| format!("{what}: populate: {e}"))?;
    Ok(Box::new(FrameRef(inner)))
}

/// Read the `frame` section of a `*.mrec` record — molrs
/// `MrecFile::open(path)?.frame()`, the inverse of [`write_mrec_frame`].
///
/// # Errors
/// An unreadable record, or one without a `frame` section (its sections are
/// named).
pub(crate) fn read_mrec_frame(path: &str) -> Result<Box<FrameRef>, String> {
    let what = "read_mrec_frame";
    let file = MrecFile::open(Path::new(path)).map_err(|e| format!("{what}: {e}"))?;
    let frame = file.frame().map_err(|e| format!("{what}: {e}"))?;
    let Some(frame) = frame else {
        let sections = file.sections().map_err(|e| format!("{what}: {e}"))?;
        return Err(format!(
            "{what}: {path} has no 'frame' section (its sections: {}); \
             a trajectory frame is read with read_mrec_trajectory_frame",
            sections.join(", ")
        ));
    };
    frame_ref_holding(frame, what)
}

/// Read frame `index` of the trajectory in a `*.mrec` record — molrs
/// `MrecReader::open(path)?.frame(index)`. Only that frame is decoded; the
/// trajectory is opened as a lazy cursor. Reads what an [`MrecWriterRef`]
/// wrote.
pub(crate) fn read_mrec_trajectory_frame(path: &str, index: u64) -> Result<Box<FrameRef>, String> {
    let what = "read_mrec_trajectory_frame";
    let frame = MrecReader::open(Path::new(path))
        .map_err(|e| format!("{what}: {e}"))?
        .frame(index)
        .map_err(|e| format!("{what}: {e}"))?
        .ok_or_else(|| format!("{what}: the trajectory has no frame {index}"))?;
    frame_ref_holding(frame, what)
}

/// The engine's streaming trajectory writer: a molrs `MrecWriter` behind
/// an opaque CXX handle. `None` once closed, so a use after close is an error
/// rather than a panic across the seam.
pub struct MrecWriterRef(Option<MrecWriter>);

fn configure_writer(
    writer: MrecWriter,
    flush_every: u64,
    durable: bool,
) -> Result<MrecWriter, String> {
    let writer = writer.with_durable(durable);
    if flush_every == 0 {
        return Ok(writer);
    }
    writer
        .with_flush_every(flush_every)
        .map_err(|e| format!("mrec_writer: {e}"))
}

/// Mint a `*.mrec` trajectory at `path`, pinned to the blocks and columns of
/// `schema_from`.
pub(crate) fn mrec_writer_create(
    path: &str,
    schema_from: &FrameRef,
    flush_every: u64,
    durable: bool,
) -> Result<Box<MrecWriterRef>, String> {
    let schema = schema_from
        .0
        .with(|frame| SequenceSchema::from_frames(std::slice::from_ref(frame)))
        .map_err(|e| format!("mrec_writer_create: {e}"))?
        .map_err(|e| format!("mrec_writer_create: {e}"))?;
    let writer =
        MrecWriter::create(path, schema).map_err(|e| format!("mrec_writer_create: {e}"))?;
    Ok(Box::new(MrecWriterRef(Some(configure_writer(
        writer,
        flush_every,
        durable,
    )?))))
}

/// Reattach to the trajectory at `path` and continue after its last committed
/// frame; whatever a crash left past the commit marker is rolled back first.
pub(crate) fn mrec_writer_open(
    path: &str,
    flush_every: u64,
    durable: bool,
) -> Result<Box<MrecWriterRef>, String> {
    let writer = MrecWriter::open(path).map_err(|e| format!("mrec_writer_open: {e}"))?;
    Ok(Box::new(MrecWriterRef(Some(configure_writer(
        writer,
        flush_every,
        durable,
    )?))))
}

/// Buffer one frame at `step` (with `time` in fs when `has_time`); a number
/// outside `0..=118` in `atoms.atomic_number` is refused, naming the row.
pub(crate) fn mrec_writer_append(
    writer: &mut MrecWriterRef,
    fref: &FrameRef,
    step: i64,
    time: f64,
    has_time: bool,
) -> Result<(), String> {
    let inner = writer
        .0
        .as_mut()
        .ok_or_else(|| "mrec_writer_append: writer is closed".to_string())?;
    let time = has_time.then_some(time);
    fref.0
        .with(|frame| {
            require_atomic_numbers(frame)?;
            inner
                .append_at(frame, step, time)
                .map_err(|e| e.to_string())
        })
        .map_err(|e| format!("mrec_writer_append: {e}"))?
        .map_err(|e| format!("mrec_writer_append: {e}"))
}

/// Commit every buffered frame (durably, unless the writer was opened with
/// `durable == false`).
pub(crate) fn mrec_writer_flush(writer: &mut MrecWriterRef) -> Result<(), String> {
    writer
        .0
        .as_mut()
        .ok_or_else(|| "mrec_writer_flush: writer is closed".to_string())?
        .flush()
        .map_err(|e| format!("mrec_writer_flush: {e}"))
}

/// Frames committed so far (0 for a closed writer).
pub(crate) fn mrec_writer_committed(writer: &MrecWriterRef) -> u64 {
    writer.0.as_ref().map_or(0, MrecWriter::committed)
}

/// Commit whatever is buffered and release the writer.
pub(crate) fn mrec_writer_close(writer: Box<MrecWriterRef>) -> Result<(), String> {
    match writer.0 {
        Some(inner) => inner.close().map_err(|e| format!("mrec_writer_close: {e}")),
        None => Ok(()),
    }
}

/// Read the first frame of an (ext)XYZ file into a materialize-ready `FrameRef`.
///
/// All parsing (atom table, `Lattice="..."` -> box) is molrs `io::read_xyz`,
/// which keeps the raw `element` column the file spells (the ExtXYZ `species`
/// property). The bridge then converts explicitly with
/// [`Element::fill_atomic_numbers`] — an unknown symbol is an error naming
/// the row, `*` is Z = 0 — and drops `element`: the result is the exact
/// boundary schema `cpu::materialize` requires
/// (`atoms.{x,y,z,atomic_number}`), and the C++ engine's frame carries no
/// raw external-format column. A caller that wants symbols back calls
/// `frame_fill_symbols`.
///
/// Z lives in `atomic_number`, not `type`: `type` is the force-field label a
/// caller owns (a String), and the vocabulary binds a key's dtype wherever it
/// appears — so writing Z there is refused outright.
pub(crate) fn read_xyz(path: &str) -> Result<Box<FrameRef>, String> {
    let what = "read_xyz";
    let mut frame =
        molrs::io::read_xyz(Path::new(path)).map_err(|e| format!("{what}: read: {e}"))?;
    Element::fill_atomic_numbers(&mut frame).map_err(|e| format!("{what}: {e}"))?;
    frame
        .get_mut("atoms")
        .expect("fill_atomic_numbers succeeded, so the atoms block exists")
        .remove(keys::ELEMENT);
    frame_ref_holding(frame, what)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::*;

    /// The frame the engine hands a writer: `atoms.{atomic_number, x, y, z}`
    /// and, when `h` is not empty, a periodic box.
    fn engine_frame(atomic_number: &[u64], x: &[f64], h: &[f64]) -> Box<FrameRef> {
        let mut fref = frame_new();
        let zeros = vec![0.0; x.len()];
        frame_set_column_u64(&mut fref, "atoms", keys::ATOMIC_NUMBER, atomic_number).unwrap();
        frame_set_column_f64(&mut fref, "atoms", "x", x).unwrap();
        frame_set_column_f64(&mut fref, "atoms", "y", &zeros).unwrap();
        frame_set_column_f64(&mut fref, "atoms", "z", &zeros).unwrap();
        if !h.is_empty() {
            frame_set_box_h(&mut fref, h).unwrap();
        }
        fref
    }

    const CUBE_10: [f64; 9] = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

    #[test]
    fn an_mrec_writer_minted_from_a_precise_frame_rounds_its_column() {
        let dir = std::env::temp_dir().join(format!(
            "molrs-cxxapi-precision-{}.mrec",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut fref = frame_new();
        frame_set_column_f64(&mut fref, "atoms", "x", &[0.123_456, 1.000_49]).unwrap();
        assert!(frame_set_precision(&mut fref, "atoms", "nope", 1e-3).is_err());
        assert!(frame_set_precision(&mut fref, "atoms", "x", 0.0).is_err());
        frame_set_precision(&mut fref, "atoms", "x", 1e-3).unwrap();
        let path = dir.to_str().unwrap();
        let mut writer = mrec_writer_create(path, &fref, 0, false).unwrap();
        mrec_writer_append(&mut writer, &fref, 0, 0.0, false).unwrap();
        mrec_writer_close(writer).unwrap();
        let back = read_mrec_trajectory_frame(path, 0).unwrap();
        assert!(read_mrec_trajectory_frame(path, 1).is_err());
        let q = 2f64.powi(-10);
        assert_eq!(
            frame_column_f64(&back, "atoms", "x").unwrap(),
            vec![(0.123_456 / q).round() * q, (1.000_49 / q).round() * q]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_xyz_writer_refuses_a_number_outside_the_schema_and_a_frame_without_one() {
        let path =
            std::env::temp_dir().join(format!("molrs-cxxapi-bad-z-{}.xyz", std::process::id()));
        let path = path.to_str().unwrap();
        let err = write_xyz(path, &engine_frame(&[1, 119], &[0.0; 2], &[]), false)
            .expect_err("Z = 119 is not an element");
        assert!(err.contains("119"), "{err}");

        let mut no_species = frame_new();
        frame_set_column_f64(&mut no_species, "atoms", "x", &[0.0]).unwrap();
        let err = write_xyz(path, &no_species, false).expect_err("no atomic_number column");
        assert!(err.contains(keys::ATOMIC_NUMBER), "{err}");
        let _ = std::fs::remove_file(path);
    }

    /// XYZ spells symbols: the writer fills `element` from `atomic_number`
    /// explicitly — on a copy, the caller's frame keeps its columns — and
    /// reading the file back through the bridge converts explicitly the other
    /// way.
    #[test]
    fn xyz_writer_spells_the_symbols_of_the_atomic_numbers() {
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-xyz-symbols-{}.xyz",
            std::process::id()
        ));
        let fref = engine_frame(&[8, 1, 1], &[0.0, 1.0, -1.0], &[]);
        write_xyz(path.to_str().unwrap(), &fref, false).unwrap();
        let mut columns = frame_block_columns(&fref, "atoms");
        columns.sort();
        assert_eq!(columns, [keys::ATOMIC_NUMBER, "x", "y", "z"]);

        let raw = molrs::io::read_xyz(path.as_path()).unwrap();
        let symbols = raw["atoms"]
            .get(keys::ELEMENT)
            .unwrap()
            .get::<String>()
            .unwrap();
        assert_eq!(symbols.iter().cloned().collect::<Vec<_>>(), ["O", "H", "H"]);
        let back = read_xyz(path.to_str().unwrap()).unwrap();
        assert_eq!(
            frame_column_u64(&back, "atoms", keys::ATOMIC_NUMBER).unwrap(),
            [8u64, 1, 1]
        );
        assert_eq!(
            frame_column_f64(&back, "atoms", "x").unwrap(),
            [0.0, 1.0, -1.0]
        );
        std::fs::remove_file(path).unwrap();
    }

    /// `append` adds a frame after the ones already in the file; `false`
    /// truncates.
    #[test]
    fn the_xyz_writer_appends_or_truncates() {
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-xyz-append-{}.xyz",
            std::process::id()
        ));
        let p = path.to_str().unwrap();
        write_xyz(p, &engine_frame(&[1], &[0.0], &[]), false).unwrap();
        write_xyz(p, &engine_frame(&[8], &[2.0], &[]), true).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text.lines().filter(|l| l.trim() == "1").count(),
            2,
            "{text}"
        );
        assert_eq!(
            frame_column_u64(&read_xyz(p).unwrap(), "atoms", keys::ATOMIC_NUMBER).unwrap(),
            [1u64]
        );
        write_xyz(p, &engine_frame(&[8], &[2.0], &[]), false).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 3, "{text}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_xyz_names_an_unknown_symbol() {
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-xyz-unknown-{}.xyz",
            std::process::id()
        ));
        std::fs::write(&path, "1\n\nXx 0 0 0\n").unwrap();
        let err = read_xyz(path.to_str().unwrap())
            .err()
            .expect("Xx is not an element symbol");
        std::fs::remove_file(path).unwrap();
        assert!(err.contains("Xx"), "{err}");
    }

    #[test]
    fn extxyz_boundary_emits_only_the_exchange_element_column() {
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-extxyz-schema-{}.xyz",
            std::process::id()
        ));
        std::fs::write(
            &path,
            concat!(
                "2\n",
                "Lattice=\"5 0 0 0 5 0 0 0 5\" ",
                "Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n",
                "O 0 0 0\n",
                "H 1 0 0\n",
            ),
        )
        .unwrap();

        let frame = read_xyz(path.to_str().unwrap()).expect("read_xyz");
        std::fs::remove_file(path).unwrap();
        let columns = frame_block_columns(&frame, "atoms");
        assert!(columns.iter().any(|column| column == "atomic_number"));
        assert!(!columns.iter().any(|column| column == "species"));
        assert!(!columns.iter().any(|column| column == "element"));
        assert!(
            !columns.iter().any(|column| column == "type"),
            "Z is `atomic_number`; `type` is the caller's force-field label"
        );
        assert_eq!(
            frame_column_u64(&frame, "atoms", "atomic_number").unwrap(),
            [8u64, 1]
        );
        assert_eq!(
            frame_box_h(&frame),
            [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]
        );
    }

    /// The core reader names the element column `element` whichever key the
    /// file used (ExtXYZ's `species`, or an `element` property), so the
    /// bridge sees one column either way.
    #[test]
    fn extxyz_boundary_reads_an_element_property_like_species() {
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-extxyz-element-key-{}.xyz",
            std::process::id()
        ));
        std::fs::write(
            &path,
            concat!(
                "1\n",
                "Lattice=\"5 0 0 0 5 0 0 0 5\" ",
                "Properties=element:S:1:pos:R:3 pbc=\"T T T\"\n",
                "H 0 0 0\n",
            ),
        )
        .unwrap();

        let frame = read_xyz(path.to_str().unwrap()).expect("read_xyz");
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            frame_column_u64(&frame, "atoms", "atomic_number").unwrap(),
            [1u64]
        );
    }

    #[test]
    fn typed_xyz_writer_preserves_numeric_metadata() {
        use crate::bridge::ffi::MetaType;
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-typed-meta-{}.xyz",
            std::process::id()
        ));
        let mut fref = engine_frame(&[1], &[0.0], &CUBE_10);
        let mut step = empty_keyed_meta_value("step".into(), MetaType::I64);
        step.i64_value = 9_007_199_254_740_993;
        frame_set_meta(&mut fref, step).unwrap();
        let mut energy = empty_keyed_meta_value("energy_eV".into(), MetaType::F64);
        energy.f64_value = -1.25;
        frame_set_meta(&mut fref, energy).unwrap();
        write_xyz(path.to_str().unwrap(), &fref, false).unwrap();

        let frame = molrs::io::read_xyz(path.as_path()).unwrap();
        assert_eq!(
            frame.meta.get("step").unwrap().as_i64(),
            Some(9_007_199_254_740_993)
        );
        assert_eq!(frame.meta.get("energy_eV").unwrap().as_f64(), Some(-1.25));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn an_mrec_frame_record_round_trips_its_columns_and_box() {
        let dir = std::env::temp_dir().join(format!(
            "molrs-cxxapi-frame-record-{}.mrec",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.to_str().unwrap();
        let mut fref = engine_frame(&[8, 1], &[0.0, 1.0], &CUBE_10);
        frame_set_column_f64(&mut fref, "atoms", "q", &[-0.8, 0.4]).unwrap();
        write_mrec_frame(path, &fref).unwrap();
        // A second write replaces the record whole.
        frame_set_column_f64(&mut fref, "atoms", "q", &[-0.6, 0.3]).unwrap();
        write_mrec_frame(path, &fref).unwrap();

        let back = read_mrec_frame(path).unwrap();
        assert_eq!(
            frame_column_u64(&back, "atoms", keys::ATOMIC_NUMBER).unwrap(),
            [8u64, 1],
            "mrec stores the canonical species field itself"
        );
        assert!(
            !frame_block_columns(&back, "atoms")
                .iter()
                .any(|column| column == keys::ELEMENT),
            "no symbol column is synthesized for the native format"
        );
        assert_eq!(frame_column_f64(&back, "atoms", "q").unwrap(), [-0.6, 0.3]);
        assert_eq!(frame_column_f64(&back, "atoms", "x").unwrap(), [0.0, 1.0]);
        assert_eq!(frame_box_h(&back), CUBE_10);
        assert!(
            read_mrec_trajectory_frame(path, 0).is_err(),
            "a frame record has no trajectory section"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_mrec_writers_refuse_a_number_outside_the_schema() {
        let dir = std::env::temp_dir().join(format!(
            "molrs-cxxapi-mrec-bad-z-{}.mrec",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.to_str().unwrap();
        let bad = engine_frame(&[1, 119], &[0.0; 2], &CUBE_10);
        let err = write_mrec_frame(path, &bad).expect_err("Z = 119 is not an element");
        assert!(err.contains("row 1") && err.contains("119"), "{err}");

        let good = engine_frame(&[1, 0], &[0.0; 2], &CUBE_10);
        let mut writer = mrec_writer_create(path, &good, 0, false).unwrap();
        mrec_writer_append(&mut writer, &good, 0, 0.0, false).unwrap();
        let err = mrec_writer_append(&mut writer, &bad, 1, 0.0, false)
            .expect_err("Z = 119 is not an element");
        assert!(err.contains("row 1") && err.contains("119"), "{err}");
        assert_eq!(mrec_writer_committed(&writer), 0);
        mrec_writer_close(writer).unwrap();
        let back = read_mrec_trajectory_frame(path, 0).unwrap();
        assert_eq!(
            frame_column_u64(&back, "atoms", keys::ATOMIC_NUMBER).unwrap(),
            [1u64, 0]
        );
        assert!(read_mrec_trajectory_frame(path, 1).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn read_mrec_frame_names_the_sections_of_a_trajectory_record() {
        let dir = std::env::temp_dir().join(format!(
            "molrs-cxxapi-no-frame-section-{}.mrec",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.to_str().unwrap();
        let fref = engine_frame(&[1], &[0.0], &[]);
        let mut writer = mrec_writer_create(path, &fref, 0, false).unwrap();
        mrec_writer_append(&mut writer, &fref, 0, 0.0, false).unwrap();
        mrec_writer_close(writer).unwrap();
        let err = read_mrec_frame(path)
            .err()
            .expect("a trajectory record has no frame section");
        assert!(err.contains("trajectory"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
