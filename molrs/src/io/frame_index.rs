//! Chunked frame indexing: where each frame of a file starts, found by
//! feeding the file in arbitrarily-sized chunks.
//!
//! This module defines the [`FrameIndexBuilder`] trait used by the streaming
//! pipeline (worker → WASM → main thread). Callers feed the source file in
//! arbitrarily-sized chunks; each implementation maintains a chunk-boundary-safe
//! state machine that emits a [`FrameOffset`] per parsed frame.
//!
//! The implementations are their formats' classes
//! ([`PdbIndexBuilder`](crate::io::pdb::PdbIndexBuilder),
//! [`XyzIndexBuilder`](crate::io::xyz::XyzIndexBuilder), the LAMMPS data and
//! dump indexers, [`SdfIndexBuilder`](crate::io::sdf::SdfIndexBuilder), and
//! the DCD, TRR and XTC ones); each byte range an indexer produces is a whole
//! one-frame file its format's `read_<fmt>` door
//! ([`read_pdb`](crate::io::read_pdb), …) reads from bytes (a DCD frame body
//! past frame 0 also through [`DcdFrameDecoder`](crate::io::dcd::DcdFrameDecoder)).
//!
//! See `docs/specs/streaming-trajectory.md` (in the molvis repo) for the
//! full design.

/// One frame's location inside the source byte stream.
///
/// `byte_offset` is the absolute byte position of the frame's first byte
/// inside the source file. `byte_len` is the number of bytes that make up
/// the frame; the slice `source[byte_offset..byte_offset + byte_len as u64]`
/// must be a self-contained input that the matching `read_<fmt>`
/// door can decode into a single [`Frame`](molrs::core::Frame).
///
/// `byte_len` is intentionally `u32`. Per-frame size is bounded — even very
/// large MD trajectories use frames in the kilobyte-to-megabyte range, and a
/// frame larger than 4 GiB is well outside the protocol's design envelope.
/// The global `byte_offset` carries `u64` so the protocol supports source
/// files up to 1 TB (well within `Number.MAX_SAFE_INTEGER` for the WASM/JS
/// bridge).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameOffset {
    /// Absolute offset (in bytes) from the start of the source.
    pub byte_offset: u64,
    /// Length, in bytes, of this frame.
    pub byte_len: u32,
}

/// Stream-friendly indexer: callers feed the raw file as zero-copy chunks
/// in source order; the indexer produces frame entries as soon as it has
/// scanned past their tail bytes.
///
/// The trait is intentionally `&mut self` rather than consuming so that
/// implementations can reuse internal scratch (line buffer, partial-frame
/// state) across calls.
///
/// Lifecycle:
///   1. Construct with `new()`.
///   2. Call [`feed`](FrameIndexBuilder::feed) repeatedly with byte chunks
///      in source order. After each `feed`, optionally call
///      [`drain`](FrameIndexBuilder::drain) to harvest any frame entries
///      that have become finalized.
///   3. After the source reaches EOF, call
///      [`finish`](FrameIndexBuilder::finish) to flush the trailing frame
///      (if any) and consume the indexer.
///
/// Implementations MUST tolerate chunks that split lines (LF or CRLF in the
/// middle), and MUST tolerate frames that span multiple chunks. After
/// `finish`, further `feed` calls panic.
pub trait FrameIndexBuilder: Send {
    /// Push the next chunk of bytes. `global_offset` is the absolute byte
    /// position of `chunk[0]` inside the source stream.
    ///
    /// Implementations MUST tolerate chunks that split lines (LF or CRLF
    /// in the middle), and MUST tolerate frames that span multiple chunks.
    fn feed(&mut self, chunk: &[u8], global_offset: u64);

    /// Drain frame entries that have been fully observed since the last
    /// `drain` (or `feed` if no prior `drain`). Successive calls are
    /// monotonic — each entry is yielded exactly once.
    fn drain(&mut self) -> Vec<FrameOffset>;

    /// Called once the source has reached EOF. Yields any trailing frame
    /// that was held back because its end wasn't yet observed, plus
    /// any frame entries still pending in the drain queue.
    ///
    /// After `finish`, the indexer is exhausted; further `feed` calls
    /// must panic.
    fn finish(self: Box<Self>) -> std::io::Result<Vec<FrameOffset>>;

    /// How many bytes have been consumed so far. Used by the worker to
    /// drive `index-progress` reports.
    fn bytes_seen(&self) -> u64;

    /// Optional known source length. Binary formats whose per-frame
    /// layout is ambiguous without the file size (DCD `has_4d` without a
    /// box) use this to resolve the layout before EOF.
    ///
    /// Default: ignore. Text indexers never need it.
    fn hint_total_bytes(&mut self, _total: u64) {}

    /// Opaque decoder state a frame's byte range needs to be decoded (DCD
    /// header + optional fixed-atom seed), as
    /// [`DcdFrameDecoder`](crate::io::dcd::DcdFrameDecoder) takes it. `None`
    /// for self-describing frames (dump, XYZ, XTC, TRR).
    fn decoder_state(&self) -> Option<Vec<u8>> {
        None
    }
}

/// Index a whole seekable stream with `builder`: feed it from the start in
/// fixed-size chunks, finish it, and put the stream back where it was.
///
/// The bridge between a chunked indexer and a file-backed
/// [`TrajectoryReader`](crate::io::reader::TrajectoryReader): a reader whose
/// format has an indexer indexes its file through here and reads frame `i`
/// from `offsets[i]` with the format's `read_<fmt>` door.
pub(crate) fn index_stream<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    mut builder: Box<dyn FrameIndexBuilder>,
) -> std::io::Result<Vec<FrameOffset>> {
    use std::io::SeekFrom;
    const CHUNK: usize = 1 << 20;
    let start = reader.stream_position()?;
    reader.seek(SeekFrom::Start(0))?;
    let mut buf = vec![0u8; CHUNK];
    let mut offset = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        builder.feed(&buf[..n], offset);
        offset += n as u64;
    }
    let offsets = builder.finish()?;
    reader.seek(SeekFrom::Start(start))?;
    Ok(offsets)
}

/// The bytes of one indexed frame of a seekable stream.
pub(crate) fn frame_bytes<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    at: FrameOffset,
) -> std::io::Result<Vec<u8>> {
    use std::io::SeekFrom;
    reader.seek(SeekFrom::Start(at.byte_offset))?;
    let mut bytes = vec![0u8; at.byte_len as usize];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Index a seekable text stream whose records each begin on a line `starts`
/// accepts (`@<TRIPOS>MOLECULE`, `data_…`), fed every line in order: one
/// [`FrameOffset`] per such line, spanning to the next one or to the end. Whatever precedes the first
/// is no record. The stream is put back where it was.
///
/// For the formats with no chunked indexer of their own, whose
/// [`TrajectoryReader`](crate::io::reader::TrajectoryReader) reads record `i`
/// as the bytes of `offsets[i]`.
pub(crate) fn index_line_records<R: std::io::BufRead + std::io::Seek>(
    reader: &mut R,
    mut starts: impl FnMut(&str) -> bool,
) -> std::io::Result<Vec<FrameOffset>> {
    use std::io::SeekFrom;
    let start = reader.stream_position()?;
    reader.seek(SeekFrom::Start(0))?;
    let mut heads = Vec::new();
    let mut offset = 0u64;
    let mut line = Vec::new();
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        if starts(&String::from_utf8_lossy(&line)) {
            heads.push(offset);
        }
        offset += n as u64;
    }
    reader.seek(SeekFrom::Start(start))?;
    let mut offsets = Vec::with_capacity(heads.len());
    for (i, &head) in heads.iter().enumerate() {
        let end = heads.get(i + 1).copied().unwrap_or(offset);
        let byte_len = u32::try_from(end - head).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("record at byte {head} is larger than 4 GiB"),
            )
        })?;
        offsets.push(FrameOffset {
            byte_offset: head,
            byte_len,
        });
    }
    Ok(offsets)
}

// ============================================================================
// Shared chunk-boundary helpers
// ============================================================================

/// Hard cap on one text line (including the unfinished carry). A
/// several-hundred-gigabyte file with no newline must not grow this
/// buffer until the process is killed — that is the original freeze
/// mode. Real dump / XYZ / PDB / SDF lines are tens of bytes.
pub(crate) const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Scratch state for "line-oriented" indexers. Every text format on the
/// streaming path is line-driven, so we factor the chunk-boundary logic
/// here.
///
/// Caller pattern:
/// ```ignore
/// let mut acc = LineAccumulator::default();
/// acc.feed(chunk, global_offset, |line, line_offset, line_len| {
///     // process one complete line; `line_offset` is the absolute byte
///     // position of the line's first byte; `line_len` is the number of
///     // bytes occupied (including any trailing \r\n or \n). The
///     // `line` slice does NOT include the trailing newline characters.
/// });
/// // ... eventually ...
/// acc.finish(|line, line_offset, line_len| { ... });
/// ```
///
/// Lines are emitted as `&str` if valid UTF-8, or `&[u8]` reinterpreted
/// via `String::from_utf8_lossy` (matching the semantics of `BufRead::read_line`).
/// We keep things in `&str` here because every format's parser is text-based.
#[derive(Default)]
pub(crate) struct LineAccumulator {
    /// Bytes carried over from a previous `feed` because the previous chunk
    /// did not end on a newline.
    carry: Vec<u8>,
    /// Absolute byte offset of `carry[0]`. Only meaningful when `!carry.is_empty()`.
    carry_offset: u64,
    /// Total bytes ever fed in (sum of all chunk lengths).
    bytes_seen: u64,
    /// Whether `finish` has been called.
    finished: bool,
    /// A line (or carry) exceeded [`MAX_LINE_BYTES`]. Further `feed`
    /// calls are ignored; [`LineAccumulator::check_line_budget`] errors.
    overflowed: bool,
}

impl LineAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes seen across all `feed` calls so far.
    pub fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }

    /// `true` after a line longer than [`MAX_LINE_BYTES`] was refused.
    #[cfg(test)]
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Error if a line exceeded the streaming budget.
    pub fn check_line_budget(&self) -> std::io::Result<()> {
        if self.overflowed {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "text trajectory line exceeds {MAX_LINE_BYTES} bytes — refuse to buffer it"
                ),
            ))
        } else {
            Ok(())
        }
    }

    fn refuse_long_line(&mut self) {
        self.overflowed = true;
        self.carry.clear();
    }

    fn emit_line<F: FnMut(&str, u64, u32)>(
        &mut self,
        raw: &[u8],
        line_offset: u64,
        total_len: u32,
        f: &mut F,
    ) {
        if total_len as usize > MAX_LINE_BYTES {
            self.refuse_long_line();
            return;
        }
        // Lossy: one bad byte must not drop the whole line (the previous
        // `from_utf8(...).unwrap_or("")` did, which hid atom rows).
        match std::str::from_utf8(raw) {
            Ok(s) => f(s, line_offset, total_len),
            Err(_) => {
                let owned = String::from_utf8_lossy(raw);
                f(&owned, line_offset, total_len);
            }
        }
    }

    /// Push the next chunk. Each complete line is delivered via `f` as
    /// `f(line_text_without_trailing_newline, line_offset, line_byte_len)`.
    /// `line_byte_len` includes the trailing `\n` or `\r\n`.
    pub fn feed<F>(&mut self, chunk: &[u8], global_offset: u64, mut f: F)
    where
        F: FnMut(&str, u64, u32),
    {
        if self.finished {
            panic!("LineAccumulator::feed called after finish");
        }
        if self.overflowed {
            return;
        }
        self.bytes_seen = global_offset.saturating_add(chunk.len() as u64);

        if chunk.is_empty() {
            return;
        }

        // If we have carry-over bytes, the "logical" buffer is carry ++ chunk.
        // We walk over the chunk only and synthesize line slices that point
        // into either `carry+chunk` (when the line spans the boundary) or
        // directly into `chunk`.
        //
        // Strategy: scan the chunk for `\n`. For each newline found at chunk
        // position `i`, the line ends at `chunk[..=i]`. If we have carry, the
        // first line is `carry ++ chunk[..=i]` — emit by buffering. Subsequent
        // lines are emitted directly from a window inside `chunk`.

        let mut start_in_chunk: usize = 0;

        for (i, &b) in chunk.iter().enumerate() {
            if b != b'\n' {
                continue;
            }
            // Line: previous-line-start .. i (inclusive of newline).
            let line_end_excl_in_chunk = i + 1;
            let line_byte_len_in_chunk = line_end_excl_in_chunk - start_in_chunk;

            if !self.carry.is_empty() && start_in_chunk == 0 {
                // First line crosses the chunk boundary: carry ++ chunk[..=i].
                let mut combined = std::mem::take(&mut self.carry);
                combined.extend_from_slice(&chunk[..line_end_excl_in_chunk]);
                let line_offset = self.carry_offset;
                let total_len = combined.len() as u32;
                let trimmed = trim_trailing_newline(&combined);
                self.emit_line(trimmed, line_offset, total_len, &mut f);
                if self.overflowed {
                    return;
                }
            } else {
                let slice = &chunk[start_in_chunk..line_end_excl_in_chunk];
                let line_offset = global_offset + start_in_chunk as u64;
                let total_len = line_byte_len_in_chunk as u32;
                let trimmed = trim_trailing_newline(slice);
                self.emit_line(trimmed, line_offset, total_len, &mut f);
                if self.overflowed {
                    return;
                }
            }
            start_in_chunk = line_end_excl_in_chunk;
        }

        // Anything between `start_in_chunk` and end-of-chunk is a partial
        // trailing line — carry it into the next feed.
        if start_in_chunk < chunk.len() {
            let add = chunk.len() - start_in_chunk;
            if self.carry.len().saturating_add(add) > MAX_LINE_BYTES {
                self.refuse_long_line();
                return;
            }
            if self.carry.is_empty() {
                // Brand-new partial line; remember its start offset.
                self.carry_offset = global_offset + start_in_chunk as u64;
            }
            // (If carry was non-empty, carry_offset is already the offset
            // of the first carry byte from a previous chunk — the boundary
            // line whose newline we haven't seen yet.)
            self.carry.extend_from_slice(&chunk[start_in_chunk..]);
        } else if !self.carry.is_empty() && start_in_chunk == 0 {
            // The chunk had zero newlines, so the entire chunk is part of
            // an existing carry-over line. Append.
            if self.carry.len().saturating_add(chunk.len()) > MAX_LINE_BYTES {
                self.refuse_long_line();
                return;
            }
            self.carry.extend_from_slice(chunk);
        }
    }

    /// Flush trailing partial line (no terminating newline) as a final line.
    /// Idempotent — calling twice is harmless (subsequent calls do nothing).
    pub fn finish<F>(&mut self, mut f: F)
    where
        F: FnMut(&str, u64, u32),
    {
        if self.finished {
            return;
        }
        self.finished = true;
        if !self.carry.is_empty() {
            let line_offset = self.carry_offset;
            let total_len = self.carry.len() as u32;
            let carry = std::mem::take(&mut self.carry);
            // The trailing partial line has no terminator, so the `trim`
            // call would be a no-op. Pass the raw slice so callers' state
            // machines see exactly what they would have seen if the file
            // had ended without a newline (the most common edge case).
            self.emit_line(&carry, line_offset, total_len, &mut f);
        }
    }
}

/// Scratch window for binary (not line-oriented) frame scanners.
///
/// Callers append sequential chunks, then ask `take_complete` to peel off
/// any prefix that a format-specific `span` function reports as one full
/// frame. The window never retains bytes of frames that have already been
/// emitted.
#[derive(Default)]
pub(crate) struct BinaryFrameScanner {
    buf: Vec<u8>,
    buf_start: u64,
    bytes_seen: u64,
    finished: bool,
    pending: Vec<FrameOffset>,
}

impl BinaryFrameScanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }

    /// Append `chunk` at `global_offset` (must be sequential with prior
    /// feeds) and emit every complete frame `span` can measure.
    ///
    /// `span` returns `Ok(None)` when the buffer is a valid prefix of a
    /// frame but not yet complete, `Ok(Some(len))` when `buf[..len]` is
    /// one frame, and `Err` on a corrupt header.
    pub fn feed<F>(&mut self, chunk: &[u8], global_offset: u64, mut span: F) -> std::io::Result<()>
    where
        F: FnMut(&[u8]) -> std::io::Result<Option<u32>>,
    {
        if self.finished {
            panic!("BinaryFrameScanner::feed called after finish");
        }
        self.append(chunk, global_offset);
        self.drain_complete(&mut span)
    }

    /// Flush remaining complete frames. A trailing incomplete prefix is
    /// dropped (same policy as the text indexers).
    pub fn finish<F>(mut self, mut span: F) -> std::io::Result<Vec<FrameOffset>>
    where
        F: FnMut(&[u8]) -> std::io::Result<Option<u32>>,
    {
        self.finished = true;
        self.drain_complete(&mut span)?;
        Ok(std::mem::take(&mut self.pending))
    }

    pub fn drain(&mut self) -> Vec<FrameOffset> {
        std::mem::take(&mut self.pending)
    }

    fn append(&mut self, chunk: &[u8], global_offset: u64) {
        if chunk.is_empty() {
            self.bytes_seen = self.bytes_seen.max(global_offset);
            return;
        }
        let expected = self.buf_start + self.buf.len() as u64;
        if self.buf.is_empty() {
            self.buf_start = global_offset;
            self.buf.extend_from_slice(chunk);
        } else if global_offset == expected {
            self.buf.extend_from_slice(chunk);
        } else if global_offset < expected {
            let skip = (expected - global_offset) as usize;
            if skip < chunk.len() {
                self.buf.extend_from_slice(&chunk[skip..]);
            }
        } else {
            // Gap: keep scanning from the new offset (corrupt / non-sequential
            // feed). Drop the orphan prefix so we do not emit a straddling
            // phantom frame.
            self.buf.clear();
            self.buf_start = global_offset;
            self.buf.extend_from_slice(chunk);
        }
        self.bytes_seen = global_offset.saturating_add(chunk.len() as u64);
    }

    fn drain_complete<F>(&mut self, span: &mut F) -> std::io::Result<()>
    where
        F: FnMut(&[u8]) -> std::io::Result<Option<u32>>,
    {
        loop {
            if self.buf.is_empty() {
                break;
            }
            match span(&self.buf)? {
                None => break,
                Some(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "binary frame span reported zero length",
                    ));
                }
                Some(len) => {
                    let n = len as usize;
                    if n > self.buf.len() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "binary frame span exceeds buffered bytes",
                        ));
                    }
                    self.pending.push(FrameOffset {
                        byte_offset: self.buf_start,
                        byte_len: len,
                    });
                    self.buf.drain(..n);
                    self.buf_start += u64::from(len);
                }
            }
        }
        Ok(())
    }
}

/// Strip a trailing `\n` or `\r\n` from `s`, returning the shorter slice.
fn trim_trailing_newline(s: &[u8]) -> &[u8] {
    if let Some((&b'\n', rest)) = s.split_last() {
        if let Some((&b'\r', rest2)) = rest.split_last() {
            rest2
        } else {
            rest
        }
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect all lines emitted from feeding `s` byte-by-byte and confirm
    /// they match a single-shot feed.
    #[test]
    fn line_accumulator_byte_by_byte_matches_single_shot() {
        let s: &[u8] = b"abc\ndef\r\nghi\n";

        let mut want: Vec<(String, u64, u32)> = Vec::new();
        let mut single = LineAccumulator::new();
        single.feed(s, 0, |line, off, len| {
            want.push((line.to_string(), off, len));
        });
        single.finish(|line, off, len| want.push((line.to_string(), off, len)));

        let mut got: Vec<(String, u64, u32)> = Vec::new();
        let mut acc = LineAccumulator::new();
        for (i, &b) in s.iter().enumerate() {
            acc.feed(&[b], i as u64, |line, off, len| {
                got.push((line.to_string(), off, len));
            });
        }
        acc.finish(|line, off, len| got.push((line.to_string(), off, len)));

        assert_eq!(got, want);
    }

    /// Trailing partial line (no newline) must be emitted from `finish`.
    #[test]
    fn line_accumulator_handles_no_trailing_newline() {
        let s: &[u8] = b"abc\ndef";
        let mut acc = LineAccumulator::new();
        let mut got: Vec<(String, u64, u32)> = Vec::new();
        acc.feed(s, 0, |line, off, len| {
            got.push((line.to_string(), off, len));
        });
        acc.finish(|line, off, len| got.push((line.to_string(), off, len)));

        assert_eq!(got, vec![("abc".into(), 0, 4), ("def".into(), 4, 3)]);
    }

    /// Chunk boundary mid-CRLF: \r at end of chunk-1, \n at start of chunk-2.
    #[test]
    fn line_accumulator_handles_crlf_split_across_chunks() {
        let s: &[u8] = b"abc\r\ndef\n";
        let mut acc = LineAccumulator::new();
        let mut got: Vec<(String, u64, u32)> = Vec::new();
        acc.feed(&s[..4], 0, |line, off, len| {
            got.push((line.to_string(), off, len));
        });
        acc.feed(&s[4..], 4, |line, off, len| {
            got.push((line.to_string(), off, len));
        });
        acc.finish(|line, off, len| got.push((line.to_string(), off, len)));

        assert_eq!(got, vec![("abc".into(), 0, 5), ("def".into(), 5, 4)]);
    }

    #[test]
    fn line_accumulator_keeps_lossy_utf8() {
        let s: &[u8] = b"ab\xFFcd\n";
        let mut acc = LineAccumulator::new();
        let mut got = Vec::new();
        acc.feed(s, 0, |line, _, _| got.push(line.to_string()));
        acc.finish(|line, _, _| got.push(line.to_string()));
        assert_eq!(got.len(), 1);
        assert!(got[0].starts_with("ab"));
        assert!(got[0].ends_with("cd"));
        assert_ne!(got[0], "", "invalid UTF-8 must not drop the line");
    }

    #[test]
    fn line_accumulator_refuses_unbounded_carry() {
        let chunk = vec![b'x'; MAX_LINE_BYTES + 1];
        let mut acc = LineAccumulator::new();
        acc.feed(&chunk, 0, |_line, _, _| panic!("must not emit"));
        assert!(acc.overflowed());
        acc.check_line_budget().unwrap_err();
    }

    fn span_fixed(n: u32) -> impl Fn(&[u8]) -> std::io::Result<Option<u32>> {
        move |buf| {
            if buf.len() < n as usize {
                Ok(None)
            } else {
                Ok(Some(n))
            }
        }
    }

    #[test]
    fn binary_scanner_chunks_match_single_shot() {
        let bytes: Vec<u8> = (0u8..40).collect();
        let mut one = BinaryFrameScanner::new();
        one.feed(&bytes, 0, span_fixed(10)).unwrap();
        let want = one.finish(span_fixed(10)).unwrap();
        assert_eq!(want.len(), 4);

        let mut acc = BinaryFrameScanner::new();
        let mut got = Vec::new();
        for (i, piece) in bytes.chunks(7).enumerate() {
            let off = (i * 7) as u64;
            acc.feed(piece, off, span_fixed(10)).unwrap();
            got.extend(acc.drain());
        }
        got.extend(acc.finish(span_fixed(10)).unwrap());
        assert_eq!(got, want);
    }

    #[test]
    fn binary_scanner_overlap_does_not_duplicate() {
        let bytes: Vec<u8> = (0u8..20).collect();
        let mut acc = BinaryFrameScanner::new();
        acc.feed(&bytes[..12], 0, span_fixed(10)).unwrap();
        let first = acc.drain();
        assert_eq!(first.len(), 1);
        acc.feed(&bytes[5..], 5, span_fixed(10)).unwrap();
        let rest = acc.finish(span_fixed(10)).unwrap();
        assert_eq!(first[0].byte_offset, 0);
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].byte_offset, 10);
    }

    #[test]
    fn binary_scanner_gap_drops_orphan_prefix() {
        let mut acc = BinaryFrameScanner::new();
        acc.feed(&[1, 2, 3, 4, 5], 0, span_fixed(10)).unwrap();
        assert!(acc.drain().is_empty());
        acc.feed(&(10u8..30).collect::<Vec<_>>(), 100, span_fixed(10))
            .unwrap();
        let got = acc.finish(span_fixed(10)).unwrap();
        assert_eq!(got[0].byte_offset, 100);
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn binary_scanner_rejects_zero_span() {
        let mut acc = BinaryFrameScanner::new();
        let err = acc.feed(&[1, 2, 3], 0, |_buf| Ok(Some(0))).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
