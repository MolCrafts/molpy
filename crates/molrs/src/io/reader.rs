use crate::core::Frame;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Result, Seek};
use std::path::Path;

/// Trait for readable and seekable buffers (used for trait objects)
pub trait ReadSeek: BufRead + Seek {}
impl<T: BufRead + Seek> ReadSeek for T {}

/// Reader for data sources returning frame-like records.
pub trait Reader {
    /// Underlying buffered reader type.
    type R: BufRead;
    /// Construct a new reader from the underlying buffered reader.
    fn new(reader: Self::R) -> Self;
}

/// Reader that yields one logical frame at a time.
///
/// # Shape
///
/// Parsing lives in a **stateless free function** per format, not in this
/// trait. [`read`](Self::read) calls it and returns a validated [`Frame`];
/// [`read_as`](Self::read_as) returns whatever type the caller wants. Multiple
/// frames are [`TrajectoryReader`]'s job, not this trait's.
pub trait FrameReader: Reader {
    /// Read one frame from the current position. `Ok(None)` on EOF.
    ///
    /// The frame is checked against the Frame schema before it is returned: a
    /// frame that violates the vocabulary is a malformed file or a reader bug,
    /// not a value to hand back as if it were fine.
    fn read(&mut self) -> Result<Option<Frame>>;

    /// Read one record as `T`.
    ///
    /// The default converts from [`read`](Self::read). A reader whose parser
    /// natively produces `T` should override this and return that value
    /// directly — `SmilesReader` parses to an [`Atomistic`], so
    /// `read_as::<Atomistic>()` hands it over instead of going
    /// `Atomistic -> Frame -> Atomistic`.
    ///
    /// The conversion adds nothing: a file reader keeps the species field its
    /// format spells, so `read_as::<Atomistic>()` on a symbol format (XYZ,
    /// PDB, SDF, …) is refused — an [`Atomistic`] needs `atomic_number`. Read
    /// the [`Frame`], convert explicitly with
    /// [`Element::fill_atomic_numbers`], then
    /// [`Atomistic::from_frame`](crate::core::Atomistic::from_frame).
    ///
    /// [`Atomistic`]: crate::core::Atomistic
    /// [`Element::fill_atomic_numbers`]: crate::core::Element::fill_atomic_numbers
    fn read_as<T: FromFrame>(&mut self) -> Result<Option<T>> {
        match self.read()? {
            Some(frame) => Ok(Some(T::from_frame(&frame)?)),
            None => Ok(None),
        }
    }
}

/// Build a value from a [`Frame`].
///
/// The target side of [`FrameReader::read_as`].
pub trait FromFrame: Sized {
    /// Convert, or explain why the frame cannot express this type.
    fn from_frame(frame: &Frame) -> Result<Self>;

    /// Fast path for readers whose parser natively produces an [`Atomistic`].
    ///
    /// Defaults to going through a frame, which is right for every type that
    /// is not itself an `Atomistic`. [`Atomistic`] overrides it to return the
    /// value unchanged, so `SmilesReader::read_as::<Atomistic>()` costs no
    /// conversion at all.
    ///
    /// This is admittedly a bulge in an otherwise frame-shaped trait. The
    /// alternative — an associated "native type" on the reader — infects every
    /// one of the twelve file readers, all of which natively produce frames,
    /// to serve the one reader that does not.
    ///
    /// [`Atomistic`]: crate::core::Atomistic
    fn from_atomistic(mol: crate::core::Atomistic) -> Result<Self> {
        let frame = mol
            .to_frame()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        Self::from_frame(&frame)
    }
}

impl FromFrame for Frame {
    fn from_frame(frame: &Frame) -> Result<Self> {
        Ok(frame.clone())
    }
}

impl FromFrame for crate::core::Atomistic {
    /// [`Atomistic::from_frame`](crate::core::Atomistic::from_frame), with no
    /// species conversion of its own: a frame without `atomic_number` is
    /// refused, naming the explicit step.
    fn from_frame(frame: &Frame) -> Result<Self> {
        crate::core::Atomistic::from_frame(frame).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{e}; an Atomistic reads 'atomic_number' only — read the Frame, run \
                     Element::fill_atomic_numbers(&mut frame), then Atomistic::from_frame"
                ),
            )
        })
    }

    fn from_atomistic(mol: crate::core::Atomistic) -> Result<Self> {
        Ok(mol)
    }
}

/// Drain a [`FrameReader`] into a `Vec`.
///
/// A free function rather than a `FrameReader` method: multi-frame access is
/// [`TrajectoryReader`]'s whole purpose. Formats that are genuinely indexable should use
/// [`TrajectoryReader::iter`]; this is for the ones that can only stream
/// forward.
pub fn collect_frames<R: FrameReader>(reader: &mut R) -> Result<Vec<Frame>> {
    let mut out = Vec::new();
    while let Some(frame) = reader.read()? {
        out.push(frame);
    }
    Ok(out)
}

/// Iterator over the frames of a [`TrajectoryReader`], from frame 0 until
/// [`TrajectoryReader::read_frame`] yields `None`.
///
/// Yields `Result<Frame>`. An `Err` leaves the cursor where it was, so the
/// iteration must be stopped on the first one rather than polled past it.
pub struct FrameIterator<'a, R: TrajectoryReader + ?Sized> {
    reader: &'a mut R,
    current: usize,
}

impl<'a, R: TrajectoryReader> Iterator for FrameIterator<'a, R> {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_frame(self.current) {
            Ok(Some(frame)) => {
                self.current += 1;
                Some(Ok(frame))
            }
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Random access by frame index over an ordered sequence of frames, whatever
/// it is stored in.
///
/// A *trajectory* here is any ordered sequence of [`Frame`]s addressed by a
/// 0-based frame index — the position in the sequence, not the MD step label a
/// frame may carry. The contract is deliberately **backend-neutral**: it
/// says nothing about files, byte offsets or seeking, so a reader over a DCD
/// file and a reader over a Zarr store (`io::mrec::zarr_storage`'s `MrecReader`) implement
/// the same three methods. That is also why [`Reader`] — which demands an
/// underlying `BufRead` — is **not** a supertrait of this one: a store-backed
/// reader has no byte stream to name.
///
/// Without that supertrait this trait is **dyn-compatible**, and callers rely
/// on it. [`Reader`] is itself dyn-incompatible (`fn new(Self::R) -> Self` takes no
/// receiver and returns `Self` by value), and a dyn-incompatible supertrait
/// makes `&mut dyn TrajectoryReader` illegal. The one method a trait object
/// cannot reach is [`iter`](Self::iter), which is `where Self: Sized`; a `dyn`
/// caller loops on [`read_frame`](Self::read_frame) until it yields `None`.
///
/// # Errors
///
/// All three required methods return `std::io::Result`. That is the lowest
/// common denominator, not a claim that every backend's failures are IO
/// failures: a backend with a richer error type flattens it in at this boundary
/// — lossy, and deliberately so, since the trait must not be captured by one
/// backend's error type. A caller who needs the structured error uses the
/// concrete reader's own doors.
pub trait TrajectoryReader {
    /// Build and cache whatever per-frame index the backend needs for random
    /// access. File readers cache byte offsets; store-backed readers cache
    /// their per-frame index arrays.
    ///
    /// Calling it is never *required* — an implementation may already have its
    /// index (a store-backed reader builds one when it opens) and answer with
    /// `Ok(())` — but a caller about to make many [`read_frame`](Self::read_frame)
    /// calls should call it once first, since a reader that scans lazily would
    /// otherwise pay for the scan on the first read.
    ///
    /// # Errors
    ///
    /// Whatever the backend raised while scanning or reading its index.
    fn build_index(&mut self) -> Result<()>;

    /// Read the frame at `index` (0-based).
    ///
    /// `Ok(None)` means `index` is past the end — the sequence has no such
    /// frame. It is a terminator, not a failure, and it is how
    /// [`FrameIterator`] knows to stop; an implementation must not return
    /// `Ok(None)` for a frame it merely could not decode.
    ///
    /// # Errors
    ///
    /// Whatever the backend raised while reading or decoding the frame.
    fn read_frame(&mut self, index: usize) -> Result<Option<Frame>>;

    /// Total number of frames the reader can serve.
    ///
    /// Every index in `0..len()` yields `Some` from
    /// [`read_frame`](Self::read_frame), and every index at or past it yields
    /// `None`. A backend that commits in batches counts only what it has
    /// committed.
    ///
    /// # Errors
    ///
    /// Whatever the backend raised while determining the count; a reader that
    /// already knows it returns `Ok` unconditionally.
    fn len(&mut self) -> Result<usize>;

    /// Whether the trajectory holds no frames at all.
    ///
    /// # Errors
    ///
    /// [`len`](Self::len)'s, which this defers to.
    fn is_empty(&mut self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Iterate frames from frame 0 until [`read_frame`](Self::read_frame) yields
    /// `None`.
    ///
    /// Each item is a `Result`. An `Err` does **not** advance the cursor and
    /// does not end the iteration, so polling past one yields the same error
    /// again: stop on the first `Err`, most simply by collecting into
    /// `Result<Vec<Frame>, _>`. Unavailable through a trait object — see the
    /// trait docs.
    fn iter(&mut self) -> FrameIterator<'_, Self>
    where
        Self: Sized,
    {
        FrameIterator {
            reader: self,
            current: 0,
        }
    }
}

/// Open a seekable file reader with automatic gzip detection based on extension.
///
/// Files with `.gz` extension are decompressed into memory to provide seekability.
///
/// # Examples
///
/// ```no_run
/// use molrs::io::reader::open_seekable;
///
/// # fn main() -> std::io::Result<()> {
/// // Opens and decompresses automatically (seekable)
/// let reader = open_seekable("data.xyz.gz")?;
///
/// // Direct read for uncompressed
/// let reader = open_seekable("data.xyz")?;
/// # Ok(())
/// # }
/// ```
/// Sequential trajectory IO buffer size (bytes).
///
/// Default `BufReader` is 8 KiB — far too small for multi‑GB TRR/XTC sequential
/// scans (one header+payload per frame). 8 MiB amortizes syscalls / NFS RTT
/// without pinning excessive RAM per open handle.
const TRAJECTORY_BUF_CAPACITY: usize = 8 * 1024 * 1024;

pub fn open_seekable<P: AsRef<Path>>(path: P) -> Result<Box<dyn ReadSeek>> {
    let path = path.as_ref();
    let file = File::open(path)?;

    if path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
    {
        // For gzipped files, decompress into memory for seekability
        use std::io::Read;
        let decoder = GzDecoder::new(file);
        let mut content = Vec::new();
        let mut buf_decoder = BufReader::with_capacity(TRAJECTORY_BUF_CAPACITY, decoder);
        buf_decoder.read_to_end(&mut content)?;
        Ok(Box::new(std::io::Cursor::new(content)))
    } else {
        Ok(Box::new(BufReader::with_capacity(
            TRAJECTORY_BUF_CAPACITY,
            file,
        )))
    }
}

/// Open a streaming file reader with automatic gzip detection based on extension.
///
/// Files with `.gz` extension are decompressed on the fly and are not seekable.
pub fn open_streaming<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>> {
    let path = path.as_ref();
    let file = File::open(path)?;

    if path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
    {
        let decoder = GzDecoder::new(file);
        Ok(Box::new(BufReader::with_capacity(
            TRAJECTORY_BUF_CAPACITY,
            decoder,
        )))
    } else {
        Ok(Box::new(BufReader::with_capacity(
            TRAJECTORY_BUF_CAPACITY,
            file,
        )))
    }
}

/// Check a freshly-read frame against the Frame schema — the read-side mirror
/// of [`check_write_frame`](crate::io::writer::check_write_frame).
///
/// Each format's frame decoder — the one function every door of the format
/// reads through — calls this before it returns a frame, so every door checks
/// once. The report names every offending column at once, so a malformed file
/// takes one round trip to diagnose rather than one per bad column.
pub fn check_read_frame<F: crate::core::FrameAccess>(frame: &F) -> Result<()> {
    crate::core::schema::Validator::canonical()
        .validate(frame)
        .map(|_| ())
        .map_err(|report| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("frame does not conform to the Frame schema:\n{report}"),
            )
        })
}

/// Parse a force-field file into a molrs
/// [`ForceField`](crate::ff::forcefield::ForceField), normalized to the
/// force-field IR (adopts the LAMMPS standard).
///
/// The one door of every force-field format's reader class
/// ([`LammpsForcefieldReader`](crate::io::lammps::LammpsForcefieldReader),
/// [`GromacsTopForcefieldReader`](crate::io::gromacs::GromacsTopForcefieldReader),
/// [`AmberPrmtopForcefieldReader`](crate::io::amber::AmberPrmtopForcefieldReader),
/// [`OpenmmXmlReader`](crate::io::openmm_xml::OpenmmXmlReader),
/// [`XmlForcefieldReader`](crate::io::XmlForcefieldReader)): the class carries
/// the format's settings (a registry, include directories, …) and
/// [`read`](Self::read) takes a [`Source`](crate::io::Source), as every frame
/// door does. A reader owns the translation from its format — element and
/// attribute names, **and unit and factor normalization** — so the resulting
/// `ForceField` needs no downstream fixup.
///
/// Reading is **total**: a malformed document or a missing required
/// attribute is an `Err`, never a silently-skipped parameter that would later
/// read as zero.
///
/// ```
/// use molrs::io::lammps::LammpsForcefieldReader;
/// use molrs::io::reader::ForceFieldReader;
///
/// let text = "special_bonds lj 0 0 0.5 coul 0 0 0.8333\n\
///             bond_style harmonic\nbond_coeff CT-HC 340.0 1.09\n";
/// let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
/// assert_eq!(ff.get_types("bond").len(), 1);
/// ```
#[cfg(feature = "ff")]
pub trait ForceFieldReader {
    /// Read the force field `src` holds: a file ([`Source::Path`](crate::io::Source::Path))
    /// or a whole file's bytes ([`Source::Bytes`](crate::io::Source::Bytes)).
    ///
    /// # Errors
    ///
    /// The frame doors' error: the OS error of a path that cannot be read,
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) for a file the format
    /// refuses, its message naming why.
    fn read<'a>(
        &self,
        src: impl Into<crate::io::Source<'a>>,
    ) -> std::io::Result<crate::ff::forcefield::ForceField>;
}

#[cfg(test)]
mod tests {
    use super::{Frame, TrajectoryReader, open_seekable, open_streaming};
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::{BufRead, Write};

    /// Number of frames the store-less reader below pretends to hold.
    const STORELESS_FRAMES: usize = 2;

    /// A [`TrajectoryReader`] backed by nothing at all — no file, no
    /// `BufRead`, no `Seek`.
    ///
    /// This is the guard for the Zarr-backed `MrecReader`: a backend whose
    /// frames come out of a store rather than a byte stream must be able to
    /// implement [`TrajectoryReader`] without an underlying reader. It
    /// deliberately does **not** `impl Reader` (which would demand
    /// `type R: BufRead` and `fn new(Self::R)` that this type cannot supply);
    /// it only becomes legal once `Reader` stops being a supertrait of
    /// [`TrajectoryReader`].
    struct StorelessReader {
        frames: usize,
    }

    impl TrajectoryReader for StorelessReader {
        fn build_index(&mut self) -> std::io::Result<()> {
            Ok(())
        }

        fn read_frame(&mut self, index: usize) -> std::io::Result<Option<Frame>> {
            if index < self.frames {
                Ok(Some(Frame::new()))
            } else {
                Ok(None)
            }
        }

        fn len(&mut self) -> std::io::Result<usize> {
            Ok(self.frames)
        }
    }

    /// `read_as::<Atomistic>()` converts no species: a symbol file is refused
    /// with the explicit step named, and that step makes it readable.
    #[test]
    fn read_as_atomistic_needs_the_explicit_species_conversion() {
        use super::FrameReader;
        use crate::core::{Atomistic, Element};
        use crate::io::xyz::XyzReader;

        let text = "2\n\nO 0 0 0\nH 0.96 0 0\n";
        let err = XyzReader::new(std::io::Cursor::new(text.as_bytes()))
            .read_as::<Atomistic>()
            .unwrap_err();
        assert!(err.to_string().contains("fill_atomic_numbers"), "{err}");

        let mut frame = XyzReader::new(std::io::Cursor::new(text.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        Element::fill_atomic_numbers(&mut frame).unwrap();
        assert_eq!(Atomistic::from_frame(&frame).unwrap().n_atoms(), 2);
    }

    #[test]
    fn open_seekable_plain_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("plain.txt");
        std::fs::write(&path, b"hello\n").expect("write temp");
        let mut reader = open_seekable(&path).expect("open seekable");
        let mut line = String::new();
        reader.read_line(&mut line).expect("read line");
        assert_eq!(line, "hello\n");
    }

    #[test]
    fn open_streaming_gz() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("data.txt.gz");
        let file = std::fs::File::create(&path).expect("create gz");
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(b"hello\n").expect("write gz");
        encoder.finish().expect("finish gz");

        let mut reader = open_streaming(&path).expect("open streaming");
        let mut line = String::new();
        reader.read_line(&mut line).expect("read line");
        assert_eq!(line, "hello\n");
    }

    #[test]
    fn trajectory_reader_needs_no_bufread_source() {
        let mut reader = StorelessReader {
            frames: STORELESS_FRAMES,
        };

        reader
            .build_index()
            .expect("build_index on a store-less reader");
        assert_eq!(reader.len().expect("len"), 2);
        assert!(reader.read_frame(0).expect("read_frame(0)").is_some());
        assert!(reader.read_frame(1).expect("read_frame(1)").is_some());
        assert!(reader.read_frame(2).expect("read_frame(2)").is_none());

        let frames: Vec<Frame> = reader
            .iter()
            .collect::<std::io::Result<Vec<Frame>>>()
            .expect("iterate a store-less reader");
        assert_eq!(frames.len(), 2);
    }
}
