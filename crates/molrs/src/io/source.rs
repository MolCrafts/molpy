//! Where frame IO reads from and writes to: [`Source`] and [`Sink`].
//!
//! Every `read_<fmt>` door and every `<Fmt>Reader::open` takes
//! `impl Into<Source>`; every `write_<fmt>` door and every
//! `<Fmt>Writer::create` takes `impl Into<Sink>`. A path is a file on disk;
//! bytes are a whole file already in memory. There is no conversion from
//! `&str`: a string is ambiguous between a path and the text of a file, so the
//! caller says which — `Path::new(s)` for a path, `s.as_bytes()` for text.

use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::io::reader::ReadSeek;

/// Where a reader reads: a file on disk, or the bytes of a whole file already
/// in memory.
///
/// A path ending in `.gz` is decompressed on the way in; bytes are read as
/// they are.
///
/// ```
/// use std::path::Path;
/// use molrs::io::{Source, read_xyz};
///
/// let text = "1\n\nO 0 0 0\n";
/// let frame = read_xyz(text.as_bytes()).unwrap();
/// assert_eq!(frame.get("atoms").unwrap().n_rows(), Some(1));
/// assert!(read_xyz(Path::new("no/such/file.xyz")).is_err());
/// assert!(matches!(Source::from(text.as_bytes()), Source::Bytes(_)));
/// ```
#[derive(Debug, Clone)]
pub enum Source<'a> {
    /// A file on disk.
    Path(&'a Path),
    /// The bytes of a whole file.
    Bytes(&'a [u8]),
}

impl<'a> From<&'a Path> for Source<'a> {
    fn from(path: &'a Path) -> Self {
        Source::Path(path)
    }
}

impl<'a> From<&'a PathBuf> for Source<'a> {
    fn from(path: &'a PathBuf) -> Self {
        Source::Path(path.as_path())
    }
}

impl<'a> From<&'a [u8]> for Source<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Source::Bytes(bytes)
    }
}

impl<'a, const N: usize> From<&'a [u8; N]> for Source<'a> {
    fn from(bytes: &'a [u8; N]) -> Self {
        Source::Bytes(bytes.as_slice())
    }
}

impl<'a> From<&'a Vec<u8>> for Source<'a> {
    fn from(bytes: &'a Vec<u8>) -> Self {
        Source::Bytes(bytes.as_slice())
    }
}

impl<'a> Source<'a> {
    /// The path, when this source is a file.
    pub(crate) fn path(&self) -> Option<&Path> {
        match self {
            Source::Path(p) => Some(p),
            Source::Bytes(_) => None,
        }
    }

    /// How an error message names this source: the path, or `bytes`.
    pub(crate) fn name(&self) -> String {
        match self.path() {
            Some(p) => p.display().to_string(),
            None => "bytes".to_owned(),
        }
    }

    /// A seekable reader over the source, borrowing its bytes.
    pub(crate) fn reader(&self) -> std::io::Result<Box<dyn ReadSeek + 'a>> {
        match self {
            Source::Bytes(b) => Ok(Box::new(Cursor::new(*b))),
            Source::Path(p) => crate::io::reader::open_seekable(p),
        }
    }

    /// A seekable reader that owns what it reads: a file stays open; bytes
    /// are copied once. What a `<Fmt>Reader::open` holds.
    pub(crate) fn owned_reader(&self) -> std::io::Result<Box<dyn ReadSeek>> {
        match self {
            Source::Bytes(b) => Ok(Box::new(Cursor::new(b.to_vec()))),
            Source::Path(p) => crate::io::reader::open_seekable(p),
        }
    }

    /// Every byte of the source.
    pub(crate) fn bytes(&self) -> std::io::Result<Cow<'a, [u8]>> {
        match self {
            Source::Bytes(b) => Ok(Cow::Borrowed(*b)),
            _ => {
                let mut out = Vec::new();
                std::io::Read::read_to_end(&mut self.reader()?, &mut out)?;
                Ok(Cow::Owned(out))
            }
        }
    }

    /// The source as UTF-8 text; `InvalidData` when it is not.
    pub(crate) fn text(&self) -> std::io::Result<Cow<'a, str>> {
        match self.bytes()? {
            Cow::Borrowed(b) => std::str::from_utf8(b)
                .map(Cow::Borrowed)
                .map_err(|e| crate::io::invalid_data(format!("{}: {e}", self.name()))),
            Cow::Owned(b) => String::from_utf8(b)
                .map(Cow::Owned)
                .map_err(|e| crate::io::invalid_data(format!("{}: {e}", self.name()))),
        }
    }
}

/// Where a writer writes: a file on disk (created, or truncated), or a buffer
/// in memory.
///
/// A buffer is cleared first and then holds exactly the bytes the file would.
///
/// ```
/// use molrs::io::{read_xyz, write_xyz};
///
/// let frame = read_xyz(b"1\n\nO 0 0 0\n").unwrap();
/// let mut out = Vec::new();
/// write_xyz(&mut out, &frame).unwrap();
/// assert!(out.starts_with(b"1\n"));
/// ```
#[derive(Debug)]
pub enum Sink<'a> {
    /// A file on disk, created or truncated.
    Path(&'a Path),
    /// A buffer, cleared and then written.
    Buffer(&'a mut Vec<u8>),
}

impl<'a> From<&'a Path> for Sink<'a> {
    fn from(path: &'a Path) -> Self {
        Sink::Path(path)
    }
}

impl<'a> From<&'a PathBuf> for Sink<'a> {
    fn from(path: &'a PathBuf) -> Self {
        Sink::Path(path.as_path())
    }
}

impl<'a> From<&'a mut Vec<u8>> for Sink<'a> {
    fn from(buffer: &'a mut Vec<u8>) -> Self {
        Sink::Buffer(buffer)
    }
}

impl<'a> Sink<'a> {
    /// Open the sink for writing: create (truncate) the file, or clear the
    /// buffer. For a streaming writer only; a single-frame door renders
    /// first ([`put_with`](Self::put_with)).
    pub(crate) fn writer(self) -> std::io::Result<SinkWriter<'a>> {
        match self {
            Sink::Path(p) => Ok(SinkWriter::File(BufWriter::new(File::create(p)?))),
            Sink::Buffer(b) => {
                b.clear();
                Ok(SinkWriter::Buffer(Cursor::new(b)))
            }
        }
    }

    /// Write `bytes` as the sink's whole content.
    pub(crate) fn put(self, bytes: &[u8]) -> std::io::Result<()> {
        let mut w = self.writer()?;
        w.write_all(bytes)?;
        w.flush()
    }

    /// Render the sink's whole content with `render`, then write it.
    ///
    /// The one door of every single-frame writer: the file is rendered in
    /// memory first, so a frame the format refuses leaves the sink as it was
    /// — an existing file is not truncated, a buffer not cleared. Only a
    /// streaming writer (one frame at a time) opens the sink before it has
    /// its content ([`writer`](Self::writer)).
    pub(crate) fn put_with(
        self,
        render: impl FnOnce(&mut Cursor<Vec<u8>>) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        let mut bytes = Cursor::new(Vec::new());
        render(&mut bytes)?;
        self.put(bytes.get_ref())
    }
}

/// An open [`Sink`]: a buffered file or a cursor over a buffer. Seekable, so a
/// writer that patches a header after the frames (DCD) writes to either.
///
/// Dropping it flushes a file and ignores the error; a writer that must report
/// one calls [`Write::flush`] itself (every `<Fmt>Writer` does, after each
/// frame).
#[derive(Debug)]
pub enum SinkWriter<'a> {
    /// A file on disk.
    File(BufWriter<File>),
    /// A buffer in memory.
    Buffer(Cursor<&'a mut Vec<u8>>),
}

impl Write for SinkWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            SinkWriter::File(w) => w.write(buf),
            SinkWriter::Buffer(w) => w.write(buf),
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            SinkWriter::File(w) => w.write_all(buf),
            SinkWriter::Buffer(w) => w.write_all(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            SinkWriter::File(w) => w.flush(),
            SinkWriter::Buffer(w) => w.flush(),
        }
    }
}

impl Seek for SinkWriter<'_> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            SinkWriter::File(w) => w.seek(pos),
            SinkWriter::Buffer(w) => w.seek(pos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_sink_is_cleared_then_written() {
        let mut buf = b"stale".to_vec();
        Sink::from(&mut buf).put(b"new").unwrap();
        assert_eq!(buf, b"new");
    }

    #[test]
    fn a_path_sink_and_a_buffer_sink_hold_the_same_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        Sink::from(path.as_path()).put(b"abc").unwrap();
        let mut buf = Vec::new();
        Sink::from(&mut buf).put(b"abc").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), buf);
        let src = Source::from(path.as_path());
        assert_eq!(&*src.bytes().unwrap(), b"abc");
        assert_eq!(&*Source::from(buf.as_slice()).text().unwrap(), "abc");
    }

    #[test]
    fn non_utf8_bytes_are_not_text() {
        let err = Source::from(&[0xffu8, 0xfe][..]).text().unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
