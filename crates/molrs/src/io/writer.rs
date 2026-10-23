//! The writer side of the io contract: [`Writer`] / [`FrameWriter`], which
//! every per-format writer class implements; [`check_write_frame`], the Frame
//! schema check each format's frame encoder runs before writing; and [`ForceFieldWriter`] /
//! [`ForceFieldWriteError`], the force-field writers' contract (feature `ff`).

use std::io::Result;
use std::io::Write;

use crate::core::Frame;

/// Generic writer for data destinations.
pub trait Writer {
    /// Underlying writer type.
    type W: Write;
    /// Construct a new writer from the underlying writer.
    fn new(writer: Self::W) -> Self;
}

/// A writer that emits one logical frame at a time.
///
/// Mirrors [`FrameReader`](crate::io::reader::FrameReader).
pub trait FrameWriter: Writer {
    /// Write one frame.
    ///
    /// The frame is checked against the Frame schema first
    /// ([`check_write_frame`], in the format's frame encoder, which the
    /// single-frame door shares). A non-conforming frame produces a file that
    /// looks fine and is wrong — the expensive kind of failure, found later by
    /// whatever reads it.
    fn write(&mut self, frame: &Frame) -> Result<()>;
}

/// Check a frame against the Frame schema before writing it — the write-side
/// mirror of [`check_read_frame`](crate::io::reader::check_read_frame).
///
/// Each format's frame encoder calls this first, so every door — the
/// single-frame `write_<fmt>` and the streaming [`FrameWriter::write`] —
/// checks once.
pub fn check_write_frame<F: crate::core::FrameAccess>(frame: &F) -> Result<()> {
    crate::core::schema::Validator::canonical()
        .validate(frame)
        .map_err(|report| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("refusing to write a frame that violates the Frame schema:\n{report}"),
            )
        })
}

/// Serialize a molrs [`ForceField`](crate::ff::forcefield::ForceField) into a
/// force-field file format.
///
/// The one door of every force-field format's writer class
/// ([`LammpsForcefieldWriter`](crate::io::lammps::LammpsForcefieldWriter),
/// [`GromacsTopForcefieldWriter`](crate::io::gromacs::GromacsTopForcefieldWriter),
/// [`AmberFrcmodWriter`](crate::io::amber::AmberFrcmodWriter),
/// [`OpenmmXmlWriter`](crate::io::openmm_xml::OpenmmXmlWriter),
/// [`XmlForcefieldWriter`](crate::io::XmlForcefieldWriter)): the class carries
/// the format's settings (type labels, precision, …) and
/// [`write`](Self::write) takes a [`Sink`](crate::io::Sink), as every frame
/// door does. Implementors own format-specific layout **and** the inverse unit
/// conversion of the matching reader.
#[cfg(feature = "ff")]
pub trait ForceFieldWriter {
    /// Write `ff` as a whole file into `sink`: a path (created or truncated)
    /// or a buffer (cleared, then holding exactly the file's bytes). Nothing
    /// is written when the format refuses `ff`.
    ///
    /// # Errors
    ///
    /// [`ForceFieldWriteError`]: what the format cannot express, by name (a
    /// typed [`IrError`](crate::ff::ir::IrError) refusal kept), or the sink's
    /// OS error.
    fn write<'a>(
        &self,
        sink: impl Into<crate::io::Sink<'a>>,
        ff: &crate::ff::forcefield::ForceField,
    ) -> std::result::Result<(), ForceFieldWriteError>;
}

/// `text` as the whole content of `sink`: the one place a force-field
/// writer's text reaches its sink.
#[cfg(feature = "ff")]
pub(crate) fn put_text(
    sink: crate::io::Sink<'_>,
    text: &str,
) -> std::result::Result<(), ForceFieldWriteError> {
    Ok(sink.put(text.as_bytes())?)
}

/// The test suites' view of [`ForceFieldWriter::write`]: what a writer
/// writes into a buffer, as text.
#[cfg(all(test, feature = "ff"))]
pub(crate) trait WrittenText: ForceFieldWriter {
    /// What [`write`](ForceFieldWriter::write) puts into a buffer for `ff`.
    fn written(
        &self,
        ff: &crate::ff::forcefield::ForceField,
    ) -> std::result::Result<String, ForceFieldWriteError> {
        let mut buffer = Vec::new();
        self.write(&mut buffer, ff)?;
        text_of(buffer)
    }
}

#[cfg(all(test, feature = "ff"))]
impl<W: ForceFieldWriter> WrittenText for W {}

/// The test suites' view of a writer method other than
/// [`write`](ForceFieldWriter::write) (`write_cmap`, `write_data_coeffs`,
/// `write_system`): what `write` puts into a buffer, as text.
#[cfg(all(test, feature = "ff"))]
pub(crate) fn written_with(
    write: impl FnOnce(&mut Vec<u8>) -> std::result::Result<(), ForceFieldWriteError>,
) -> std::result::Result<String, ForceFieldWriteError> {
    let mut buffer = Vec::new();
    write(&mut buffer)?;
    text_of(buffer)
}

/// A force-field writer's buffer as the text it is; the writers write UTF-8
/// only.
#[cfg(all(test, feature = "ff"))]
pub(crate) fn text_of(buffer: Vec<u8>) -> std::result::Result<String, ForceFieldWriteError> {
    String::from_utf8(buffer).map_err(|e| ForceFieldWriteError::from(crate::io::invalid_data(e)))
}

/// Why a writer wrote nothing.
///
/// An engine that cannot hold a style refuses it with a typed
/// [`IrError::NoEngineForm`](crate::ff::ir::IrError::NoEngineForm) (`ff-ir-02-protocol` §8), kept here so a caller
/// matches on the refusal ([`ir`](Self::ir)) instead of reading a message —
/// a binding raises `molrs.ff.ir.NoEngineForm` from it. Anything else (a
/// parameter the format has no field for, a malformed row) is the message
/// alone. Either way the error reads as its message: it dereferences to
/// `str`.
///
/// It meets the frame doors' [`std::io::Error`] both ways: a sink that fails
/// is `From<std::io::Error>` (its [`kind`](Self::kind) kept), and the error
/// converts into a `std::io::Error` — [`kind`](Self::kind), with this error
/// as its inner error, so the refusal is still there to downcast.
///
/// ```
/// use molrs::ff::forcefield::ForceField;
/// use molrs::ff::ir::Params;
/// use molrs::io::writer::ForceFieldWriter;
/// use molrs::io::gromacs::GromacsTopForcefieldWriter;
/// use molrs::ff::ir::IrError;
///
/// let mut ff = ForceField::new("t");
/// let mut style = Params::new();
/// style.set("expression", "k*(r-r0)^4");
/// ff.def_style("bond", "quartic", style).unwrap();
/// let err = GromacsTopForcefieldWriter::new()
///     .write(&mut Vec::<u8>::new(), &ff)
///     .unwrap_err();
/// assert!(matches!(
///     err.ir(),
///     Some(IrError::NoEngineForm { engine, style, .. }) if engine == "GROMACS" && style == "quartic"
/// ));
/// assert!(err.contains("GROMACS has no form for bond `quartic`"));
/// ```
#[cfg(feature = "ff")]
#[derive(Clone, Debug, PartialEq)]
pub struct ForceFieldWriteError {
    message: String,
    refusal: Option<Box<crate::ff::ir::IrError>>,
    kind: std::io::ErrorKind,
}

#[cfg(feature = "ff")]
impl ForceFieldWriteError {
    /// The force-field IR's refusal, when this is one.
    pub fn ir(&self) -> Option<&crate::ff::ir::IrError> {
        self.refusal.as_deref()
    }

    /// [`InvalidData`](std::io::ErrorKind::InvalidData) for what the format
    /// refuses, the OS error's kind for a sink that failed.
    pub fn kind(&self) -> std::io::ErrorKind {
        self.kind
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The same error, its message prefixed with `what` (`"bonds label
    /// `CT-HC`: …"`); a refusal stays typed.
    pub fn context(mut self, what: impl std::fmt::Display) -> Self {
        self.message = format!("{what}: {}", self.message);
        self
    }
}

#[cfg(feature = "ff")]
impl std::ops::Deref for ForceFieldWriteError {
    type Target = str;

    fn deref(&self) -> &str {
        &self.message
    }
}

#[cfg(feature = "ff")]
impl std::fmt::Display for ForceFieldWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

#[cfg(feature = "ff")]
impl std::error::Error for ForceFieldWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.refusal
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

#[cfg(feature = "ff")]
impl From<crate::ff::ir::IrError> for ForceFieldWriteError {
    fn from(e: crate::ff::ir::IrError) -> Self {
        Self {
            message: e.to_string(),
            refusal: Some(Box::new(e)),
            kind: std::io::ErrorKind::InvalidData,
        }
    }
}

#[cfg(feature = "ff")]
impl From<String> for ForceFieldWriteError {
    fn from(message: String) -> Self {
        Self {
            message,
            refusal: None,
            kind: std::io::ErrorKind::InvalidData,
        }
    }
}

#[cfg(feature = "ff")]
impl From<&str> for ForceFieldWriteError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

#[cfg(feature = "ff")]
impl From<std::io::Error> for ForceFieldWriteError {
    fn from(e: std::io::Error) -> Self {
        Self {
            message: e.to_string(),
            refusal: None,
            kind: e.kind(),
        }
    }
}

#[cfg(feature = "ff")]
impl From<ForceFieldWriteError> for std::io::Error {
    fn from(e: ForceFieldWriteError) -> Self {
        std::io::Error::new(e.kind, e)
    }
}

#[cfg(feature = "ff")]
impl From<ForceFieldWriteError> for String {
    fn from(e: ForceFieldWriteError) -> Self {
        e.message
    }
}

/// Capture the bytes a writer emits for assertions in the unit suites.
#[cfg(test)]
pub(crate) fn capture<E: From<std::io::Error>>(
    write: impl FnOnce(&mut Vec<u8>) -> std::result::Result<(), E>,
) -> std::result::Result<String, E> {
    let mut bytes = Vec::new();
    write(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e).into())
}
