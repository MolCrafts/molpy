//! GROMACS XTC binary trajectory reader and writer.
use crate::core::Cell;
use crate::core::unit_factors::NM_TO_ANGSTROM;
use crate::io::frame_columns::insert_column;
use crate::io::frame_index::{BinaryFrameScanner, FrameIndexBuilder, FrameOffset};
use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::writer::{FrameWriter, Writer};
use crate::io::xdr;
use crate::io::{Sink, SinkWriter, Source, invalid_data};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::FrameAccess;
use molrs::core::SimBox;
use molrs::op::{F, Idx};
use ndarray::{Array1, IxDyn};
use std::io::{BufRead, Read, Result, Seek, SeekFrom, Write};
use std::sync::OnceLock;

/// Classic XTC magic number.
const XTC_MAGIC: i32 = 1995;
/// Forward-magic variant exercised by `ubiquitin_faux2023magic.xtc`.
const XTC_MAGIC_2023: i32 = 2023;
const DIM: usize = 3;
const FIRSTIDX: i32 = 9;

/// Bit-width selector table for the small-integer encoding.
const MAGICINTS: [i32; 73] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 101, 128, 161, 203, 256,
    322, 406, 512, 645, 812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321,
    13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031, 131072, 165140, 208063,
    262144, 330280, 416127, 524287, 660561, 832255, 1048576, 1321122, 1664510, 2097152, 2642245,
    3329021, 4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
];
/// Largest valid index into [`MAGICINTS`].
const LASTIDX: i32 = MAGICINTS.len() as i32 - 1;

fn unsupported<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, e.to_string())
}

// ---------------------------------------------------------------------------
// Bit-width helpers
// ---------------------------------------------------------------------------

/// Number of bits needed to store an integer in `[0, size)`.
fn sizeofint(size: u32) -> i32 {
    let mut num: u32 = 1;
    let mut nbits = 0i32;
    while size >= num && nbits < 32 {
        nbits += 1;
        num = num.wrapping_shl(1);
    }
    nbits
}

/// Number of bits needed to store `n_ints` integers `< sizes[i]` as a
/// single mixed-radix number.
fn sizeofints(n_ints: usize, sizes: &[u32; 3]) -> i32 {
    let mut bytes = [0u32; 32];
    let mut n_bytes = 1usize;
    bytes[0] = 1;
    for &size in sizes.iter().take(n_ints) {
        let mut tmp = 0u64;
        let mut bytecnt = 0usize;
        while bytecnt < n_bytes {
            tmp += bytes[bytecnt] as u64 * size as u64;
            bytes[bytecnt] = (tmp & 0xff) as u32;
            tmp >>= 8;
            bytecnt += 1;
        }
        while tmp != 0 {
            bytes[bytecnt] = (tmp & 0xff) as u32;
            bytecnt += 1;
            tmp >>= 8;
        }
        n_bytes = bytecnt;
    }
    let last = n_bytes - 1;
    let mut num = 1u32;
    let mut nbits = 0i32;
    while bytes[last] >= num {
        nbits += 1;
        num = num.wrapping_mul(2);
    }
    nbits + (last as i32) * 8
}

// ---------------------------------------------------------------------------
// Bit reader (decode)
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    buf: &'a [u8],
    cnt: usize,
    lastbits: i32,
    lastbyte: u32,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            cnt: 0,
            lastbits: 0,
            lastbyte: 0,
        }
    }

    fn next_byte(&mut self) -> Result<u32> {
        let b = *self
            .buf
            .get(self.cnt)
            .ok_or_else(|| invalid_data("XTC compressed buffer underrun"))?;
        self.cnt += 1;
        Ok(b as u32)
    }

    /// Receive `n_bits` bits, MSB-first.
    fn receivebits(&mut self, mut n_bits: i32) -> Result<i32> {
        let mut num: u32 = 0;
        let mask = if n_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << n_bits) - 1
        };
        while n_bits >= 8 {
            self.lastbyte = (self.lastbyte << 8) | self.next_byte()?;
            num |= (self.lastbyte >> self.lastbits) << (n_bits - 8);
            n_bits -= 8;
        }
        if n_bits > 0 {
            if self.lastbits < n_bits {
                self.lastbits += 8;
                self.lastbyte = (self.lastbyte << 8) | self.next_byte()?;
            }
            self.lastbits -= n_bits;
            num |= (self.lastbyte >> self.lastbits) & ((1u32 << n_bits) - 1);
        }
        num &= mask;
        Ok(num as i32)
    }

    /// Receive `n_ints` integers packed with `n_bits` total bits.
    fn receiveints(
        &mut self,
        n_ints: usize,
        mut n_bits: i32,
        sizes: &[u32; 3],
        nums: &mut [i32; 3],
    ) -> Result<()> {
        let mut bytes = [0i32; 32];
        let mut n_bytes = 0usize;
        while n_bits > 8 {
            bytes[n_bytes] = self.receivebits(8)?;
            n_bytes += 1;
            n_bits -= 8;
        }
        if n_bits > 0 {
            bytes[n_bytes] = self.receivebits(n_bits)?;
            n_bytes += 1;
        }
        for i in (1..n_ints).rev() {
            let mut num = 0u32;
            for j in (0..n_bytes).rev() {
                num = (num << 8) | bytes[j] as u32;
                let p = num / sizes[i];
                bytes[j] = p as i32;
                num -= p * sizes[i];
            }
            nums[i] = num as i32;
        }
        nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[inline]
fn check_idx(idx: i32) -> Result<usize> {
    if !(0..=LASTIDX).contains(&idx) {
        return Err(invalid_data(format!("XTC smallidx {idx} out of range")));
    }
    Ok(idx as usize)
}

/// Decompress a frame's coordinate bitstream into an interleaved `[x,y,z,…]`
/// nm vector of length `3*natoms`.
fn decompress_coords(
    bytes: &[u8],
    natoms: usize,
    precision: f32,
    minint: [i32; 3],
    maxint: [i32; 3],
    smallidx_init: i32,
) -> Result<Vec<f64>> {
    let inv: f32 = if precision != 0.0 {
        1.0 / precision
    } else {
        1.0
    };
    let mut sizeint = [0u32; 3];
    for d in 0..DIM {
        if maxint[d] < minint[d] {
            return Err(invalid_data("XTC maxint < minint"));
        }
        sizeint[d] = (maxint[d] - minint[d]) as u32 + 1;
    }

    let (bitsize, bitsizeint) = if (sizeint[0] | sizeint[1] | sizeint[2]) > 0x00ff_ffff {
        (
            0,
            [
                sizeofint(sizeint[0]),
                sizeofint(sizeint[1]),
                sizeofint(sizeint[2]),
            ],
        )
    } else {
        (sizeofints(DIM, &sizeint), [0i32; 3])
    };

    let mut smallidx = smallidx_init;
    let init_smaller_idx = FIRSTIDX.max(smallidx - 1);
    let mut smaller = MAGICINTS[check_idx(init_smaller_idx)?] / 2;
    let mut smallnum = MAGICINTS[check_idx(smallidx)?] / 2;
    let mut sizesmall = [MAGICINTS[check_idx(smallidx)?] as u32; 3];

    let mut br = BitReader::new(bytes);
    let mut out: Vec<f64> = Vec::with_capacity(natoms * DIM);
    let emit = |out: &mut Vec<f64>, c: &[i32; 3]| {
        out.push((c[0] as f32 * inv) as f64);
        out.push((c[1] as f32 * inv) as f64);
        out.push((c[2] as f32 * inv) as f64);
    };

    let mut i = 0usize;
    // `run` persists across atoms: a `flag == 0` bit means "same run length as
    // the previous atom", so it must NOT be reset each iteration.
    let mut run = 0i32;
    while i < natoms {
        let mut thiscoord = [0i32; 3];
        if bitsize == 0 {
            thiscoord[0] = br.receivebits(bitsizeint[0])?;
            thiscoord[1] = br.receivebits(bitsizeint[1])?;
            thiscoord[2] = br.receivebits(bitsizeint[2])?;
        } else {
            br.receiveints(DIM, bitsize, &sizeint, &mut thiscoord)?;
        }
        i += 1;
        thiscoord[0] += minint[0];
        thiscoord[1] += minint[1];
        thiscoord[2] += minint[2];
        let mut prevcoord = thiscoord;

        let flag = br.receivebits(1)?;
        let mut is_smaller = 0i32;
        if flag == 1 {
            run = br.receivebits(5)?;
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller -= 1;
        }
        if run > 0 {
            if i + (run as usize / DIM) > natoms {
                return Err(invalid_data("XTC run length exceeds atom count"));
            }
            let mut k = 0i32;
            while k < run {
                let mut rc = [0i32; 3];
                br.receiveints(DIM, smallidx, &sizesmall, &mut rc)?;
                i += 1;
                rc[0] += prevcoord[0] - smallnum;
                rc[1] += prevcoord[1] - smallnum;
                rc[2] += prevcoord[2] - smallnum;
                if k == 0 {
                    std::mem::swap(&mut rc, &mut prevcoord);
                    emit(&mut out, &prevcoord);
                } else {
                    prevcoord = rc;
                }
                emit(&mut out, &rc);
                k += 3;
            }
        } else {
            emit(&mut out, &thiscoord);
        }

        smallidx += is_smaller;
        if is_smaller < 0 {
            smallnum = smaller;
            smaller = if smallidx > FIRSTIDX {
                MAGICINTS[check_idx(smallidx - 1)?] / 2
            } else {
                0
            };
        } else if is_smaller > 0 {
            smaller = smallnum;
            smallnum = MAGICINTS[check_idx(smallidx)?] / 2;
        }
        sizesmall = [MAGICINTS[check_idx(smallidx)?] as u32; 3];
    }

    if out.len() != natoms * DIM {
        return Err(invalid_data(format!(
            "XTC decode produced {} coords, expected {}",
            out.len(),
            natoms * DIM
        )));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Bit writer (encode)
// ---------------------------------------------------------------------------

struct BitWriter {
    data: Vec<u8>,
    cnt: usize,
    lastbits: i32,
    lastbyte: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            cnt: 0,
            lastbits: 0,
            lastbyte: 0,
        }
    }

    fn put(&mut self, idx: usize, val: u8) {
        if idx >= self.data.len() {
            self.data.resize(idx + 1, 0);
        }
        self.data[idx] = val;
    }

    fn sendbits(&mut self, mut n_bits: i32, num: u32) {
        let mut cnt = self.cnt;
        let mut lastbyte = self.lastbyte;
        let mut lastbits = self.lastbits;
        while n_bits >= 8 {
            lastbyte = (lastbyte << 8) | ((num >> (n_bits - 8)) & 0xff);
            self.put(cnt, (lastbyte >> lastbits) as u8);
            cnt += 1;
            n_bits -= 8;
        }
        if n_bits > 0 {
            let mask = (1u32 << n_bits) - 1;
            lastbyte = (lastbyte << n_bits) | (num & mask);
            lastbits += n_bits;
            if lastbits >= 8 {
                lastbits -= 8;
                self.put(cnt, (lastbyte >> lastbits) as u8);
                cnt += 1;
            }
        }
        self.cnt = cnt;
        self.lastbyte = lastbyte;
        self.lastbits = lastbits;
        if lastbits > 0 {
            self.put(cnt, (lastbyte << (8 - lastbits)) as u8);
        }
    }

    fn sendints(&mut self, n_ints: usize, n_bits: i32, sizes: &[u32; 3], nums: &[u32; 3]) {
        let mut bytes = [0u32; 32];
        let mut tmp = nums[0];
        let mut n_bytes = 0usize;
        loop {
            bytes[n_bytes] = tmp & 0xff;
            n_bytes += 1;
            tmp >>= 8;
            if tmp == 0 {
                break;
            }
        }
        for i in 1..n_ints {
            let mut carry = nums[i] as u64;
            let mut bytecnt = 0usize;
            while bytecnt < n_bytes {
                carry += bytes[bytecnt] as u64 * sizes[i] as u64;
                bytes[bytecnt] = (carry & 0xff) as u32;
                carry >>= 8;
                bytecnt += 1;
            }
            while carry != 0 {
                bytes[bytecnt] = (carry & 0xff) as u32;
                bytecnt += 1;
                carry >>= 8;
            }
            n_bytes = bytecnt;
        }
        if n_bits >= (n_bytes as i32) * 8 {
            for &b in bytes.iter().take(n_bytes) {
                self.sendbits(8, b);
            }
            self.sendbits(n_bits - (n_bytes as i32) * 8, 0);
        } else {
            for &b in bytes.iter().take(n_bytes - 1) {
                self.sendbits(8, b);
            }
            self.sendbits(n_bits - ((n_bytes - 1) as i32) * 8, bytes[n_bytes - 1]);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        let mut nbytes = self.cnt;
        if self.lastbits != 0 {
            nbytes += 1;
        }
        if self.data.len() < nbytes {
            self.data.resize(nbytes, 0);
        }
        self.data.truncate(nbytes);
        self.data
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// `(minint, maxint, smallidx, compressed_bytes)` — the pieces the XTC writer
/// emits after the precision field.
type Compressed = ([i32; 3], [i32; 3], i32, Vec<u8>);

/// Compress an interleaved `[x,y,z,…]` nm coordinate vector.
fn compress_coords(coords: &[f64], natoms: usize, precision: f32) -> Result<Compressed> {
    let size = natoms;
    let maxabs = (i32::MAX - 2) as f32;

    let mut ip = vec![0i32; size * DIM];
    let mut minint = [i32::MAX; 3];
    let mut maxint = [i32::MIN; 3];
    // i64 so large coordinate spreads (e.g. `large_diff.xtc`) don't overflow.
    let mut mindiff = i64::MAX;
    let mut oldlint = [0i32; 3];
    for a in 0..size {
        let mut lint = [0i32; 3];
        for d in 0..DIM {
            let val = coords[a * DIM + d] as f32;
            let lf = if val >= 0.0 {
                val * precision + 0.5
            } else {
                val * precision - 0.5
            };
            if lf.abs() > maxabs {
                return Err(unsupported("XTC: coordinate too large for compression"));
            }
            let li = lf as i32;
            lint[d] = li;
            if li < minint[d] {
                minint[d] = li;
            }
            if li > maxint[d] {
                maxint[d] = li;
            }
            ip[a * DIM + d] = li;
        }
        if a > 0 {
            let diff = (oldlint[0] as i64 - lint[0] as i64).abs()
                + (oldlint[1] as i64 - lint[1] as i64).abs()
                + (oldlint[2] as i64 - lint[2] as i64).abs();
            if diff < mindiff {
                mindiff = diff;
            }
        }
        oldlint = lint;
    }

    let mut sizeint = [0u32; 3];
    for d in 0..DIM {
        sizeint[d] = (maxint[d] as i64 - minint[d] as i64) as u32 + 1;
    }
    let (bitsize, bitsizeint) = if (sizeint[0] | sizeint[1] | sizeint[2]) > 0x00ff_ffff {
        (
            0,
            [
                sizeofint(sizeint[0]),
                sizeofint(sizeint[1]),
                sizeofint(sizeint[2]),
            ],
        )
    } else {
        (sizeofints(DIM, &sizeint), [0i32; 3])
    };

    let mut smallidx = FIRSTIDX;
    while smallidx < LASTIDX && (MAGICINTS[(smallidx + 1) as usize] as i64) < mindiff {
        smallidx += 1;
    }
    let smallidx_initial = smallidx;
    let maxidx = LASTIDX.min(smallidx + 8);
    let minidx = maxidx - 8;
    let mut smaller = MAGICINTS[FIRSTIDX.max(smallidx - 1) as usize] / 2;
    let mut smallnum = MAGICINTS[smallidx as usize] / 2;
    let mut sizesmall = [MAGICINTS[smallidx as usize] as u32; 3];
    let larger = MAGICINTS[maxidx as usize];

    let mut bw = BitWriter::new();
    let mut prevcoord = [0i32; 3];
    let mut prevrun = -1i32;
    let mut i = 0usize;
    while i < size {
        let tc = i * DIM;
        // i64 deltas avoid overflow on large-spread coordinates.
        let adelta = |a: i32, b: i32| (a as i64 - b as i64).abs();
        let mut is_small = 0i32;
        let larger = larger as i64;
        let smallnum_i = smallnum as i64;
        let mut is_smaller = if smallidx < maxidx
            && i >= 1
            && adelta(ip[tc], prevcoord[0]) < larger
            && adelta(ip[tc + 1], prevcoord[1]) < larger
            && adelta(ip[tc + 2], prevcoord[2]) < larger
        {
            1
        } else if smallidx > minidx {
            -1
        } else {
            0
        };

        if i + 1 < size
            && adelta(ip[tc], ip[tc + 3]) < smallnum_i
            && adelta(ip[tc + 1], ip[tc + 4]) < smallnum_i
            && adelta(ip[tc + 2], ip[tc + 5]) < smallnum_i
        {
            ip.swap(tc, tc + 3);
            ip.swap(tc + 1, tc + 4);
            ip.swap(tc + 2, tc + 5);
            is_small = 1;
        }

        let tmpcoord = [
            (ip[tc] as i64 - minint[0] as i64) as u32,
            (ip[tc + 1] as i64 - minint[1] as i64) as u32,
            (ip[tc + 2] as i64 - minint[2] as i64) as u32,
        ];
        if bitsize == 0 {
            bw.sendbits(bitsizeint[0], tmpcoord[0]);
            bw.sendbits(bitsizeint[1], tmpcoord[1]);
            bw.sendbits(bitsizeint[2], tmpcoord[2]);
        } else {
            bw.sendints(DIM, bitsize, &sizeint, &tmpcoord);
        }
        prevcoord = [ip[tc], ip[tc + 1], ip[tc + 2]];
        let mut tc2 = tc + DIM;
        i += 1;

        let mut run = 0i32;
        let mut runbuf = [0u32; 30];
        if is_small == 0 && is_smaller == -1 {
            is_smaller = 0;
        }
        while is_small != 0 && run < 8 * 3 {
            if is_smaller == -1 {
                let d0 = (ip[tc2] - prevcoord[0]) as i64;
                let d1 = (ip[tc2 + 1] - prevcoord[1]) as i64;
                let d2 = (ip[tc2 + 2] - prevcoord[2]) as i64;
                if d0 * d0 + d1 * d1 + d2 * d2 >= (smaller as i64) * (smaller as i64) {
                    is_smaller = 0;
                }
            }
            runbuf[run as usize] = (ip[tc2] as i64 - prevcoord[0] as i64 + smallnum as i64) as u32;
            run += 1;
            runbuf[run as usize] =
                (ip[tc2 + 1] as i64 - prevcoord[1] as i64 + smallnum as i64) as u32;
            run += 1;
            runbuf[run as usize] =
                (ip[tc2 + 2] as i64 - prevcoord[2] as i64 + smallnum as i64) as u32;
            run += 1;
            prevcoord = [ip[tc2], ip[tc2 + 1], ip[tc2 + 2]];
            i += 1;
            tc2 += DIM;
            is_small = 0;
            if i < size
                && adelta(ip[tc2], prevcoord[0]) < smallnum_i
                && adelta(ip[tc2 + 1], prevcoord[1]) < smallnum_i
                && adelta(ip[tc2 + 2], prevcoord[2]) < smallnum_i
            {
                is_small = 1;
            }
        }

        if run != prevrun || is_smaller != 0 {
            prevrun = run;
            bw.sendbits(1, 1);
            bw.sendbits(5, (run + is_smaller + 1) as u32);
        } else {
            bw.sendbits(1, 0);
        }
        let mut k = 0usize;
        while (k as i32) < run {
            let chunk = [runbuf[k], runbuf[k + 1], runbuf[k + 2]];
            bw.sendints(DIM, smallidx, &sizesmall, &chunk);
            k += DIM;
        }
        if is_smaller != 0 {
            smallidx += is_smaller;
            if is_smaller > 0 {
                smaller = smallnum;
                smallnum = MAGICINTS[smallidx as usize] / 2;
            } else {
                smallnum = smaller;
                smaller = MAGICINTS[(smallidx - 1) as usize] / 2;
            }
            sizesmall = [MAGICINTS[smallidx as usize] as u32; 3];
        }
    }

    Ok((minint, maxint, smallidx_initial, bw.finish()))
}

// ---------------------------------------------------------------------------
// Frame parsing
// ---------------------------------------------------------------------------

/// Parsed XTC per-frame header (everything up to the coordinate block).
#[derive(Debug, Clone)]
struct XtcHeader {
    natoms: usize,
    step: i32,
    time: f32,
    boxv: [f32; 9],
    /// The 2023 magic stores the compressed byte count as a 64-bit integer
    /// (GROMACS' fix for frames whose compressed size exceeds 2 GiB).
    wide_nbytes: bool,
}

/// Read the compressed-buffer byte count (`i64` for the 2023 format, else
/// `i32`).
fn read_nbytes<R: Read>(r: &mut R, wide: bool) -> Result<usize> {
    let nbytes = if wide {
        xdr::decode_i64(r)?
    } else {
        xdr::decode_i32(r)? as i64
    };
    if nbytes < 0 {
        return Err(invalid_data(format!("negative XTC nbytes {nbytes}")));
    }
    Ok(nbytes as usize)
}

fn read_header<R: Read>(r: &mut R) -> Result<XtcHeader> {
    let magic = xdr::decode_i32(r)?;
    if magic != XTC_MAGIC && magic != XTC_MAGIC_2023 {
        return Err(invalid_data(format!(
            "bad XTC magic {magic} (expected {XTC_MAGIC} or {XTC_MAGIC_2023})"
        )));
    }
    let natoms = xdr::decode_i32(r)?;
    if natoms <= 0 {
        return Err(invalid_data(format!("invalid_data XTC natoms {natoms}")));
    }
    let step = xdr::decode_i32(r)?;
    let time = xdr::decode_f32(r)?;
    let mut boxv = [0f32; 9];
    for b in boxv.iter_mut() {
        *b = xdr::decode_f32(r)?;
    }
    Ok(XtcHeader {
        natoms: natoms as usize,
        step,
        time,
        boxv,
        wide_nbytes: magic == XTC_MAGIC_2023,
    })
}

/// Read and decompress the coordinate block. Returns `(coords, precision)`.
fn read_coords<R: Read>(r: &mut R, natoms: usize, wide_nbytes: bool) -> Result<(Vec<f64>, f32)> {
    let size = xdr::decode_i32(r)?;
    if size as usize != natoms {
        return Err(invalid_data(format!(
            "XTC coord size {size} disagrees with header natoms {natoms}"
        )));
    }
    if natoms <= 9 {
        let mut out = Vec::with_capacity(natoms * DIM);
        for _ in 0..natoms * DIM {
            out.push(xdr::decode_f32(r)? as f64);
        }
        return Ok((out, 0.0));
    }
    let precision = xdr::decode_f32(r)?;
    let mut minint = [0i32; 3];
    for m in minint.iter_mut() {
        *m = xdr::decode_i32(r)?;
    }
    let mut maxint = [0i32; 3];
    for m in maxint.iter_mut() {
        *m = xdr::decode_i32(r)?;
    }
    let smallidx = xdr::decode_i32(r)?;
    let nbytes = read_nbytes(r, wide_nbytes)?;
    let buf = xdr::decode_opaque(r, nbytes)?;
    let coords = decompress_coords(&buf, natoms, precision, minint, maxint, smallidx)?;
    Ok((coords, precision))
}

fn build_simbox(boxv: &[f32; 9]) -> Option<Result<SimBox>> {
    if boxv.iter().all(|&v| v == 0.0) {
        return None;
    }
    // Row-stored vectors → column-stored H: H[r][c] = box[c*3 + r].
    let h = std::array::from_fn(|r| std::array::from_fn(|c| boxv[c * DIM + r] as F));
    Some(
        SimBox::new(Cell::Matrix(h), [0.0; 3], [true; 3])
            .map_err(|e| invalid_data(format!("XTC box: {e:?}"))),
    )
}

/// Parse one XTC frame at the current position (no seek). `Ok(None)` on EOF.
fn parse_frame_here<R: Read>(r: &mut R) -> Result<Option<Frame>> {
    let hdr = match read_header(r) {
        Ok(h) => h,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    };
    let natoms = hdr.natoms;
    let (coords, precision) = read_coords(r, natoms, hdr.wide_nbytes)?;

    let mut atoms = Block::new();
    let id_arr = Array1::from_iter(1..=natoms as Idx)
        .into_shape_with_order(IxDyn(&[natoms]))
        .map_err(invalid_data)?;
    atoms.insert("id", id_arr, None).map_err(invalid_data)?;
    let mut x = Vec::with_capacity(natoms);
    let mut y = Vec::with_capacity(natoms);
    let mut z = Vec::with_capacity(natoms);
    for a in 0..natoms {
        x.push(coords[a * DIM] as F * NM_TO_ANGSTROM.get());
        y.push(coords[a * DIM + 1] as F * NM_TO_ANGSTROM.get());
        z.push(coords[a * DIM + 2] as F * NM_TO_ANGSTROM.get());
    }
    insert_column(&mut atoms, "x", x)?;
    insert_column(&mut atoms, "y", y)?;
    insert_column(&mut atoms, "z", z)?;

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.simbox = match build_simbox(&hdr.boxv) {
        Some(res) => {
            let sb = res?;
            let scale = NM_TO_ANGSTROM.get();
            let h = sb.matrix().map(|row| row.map(|v| v * scale));
            let origin = sb.origin().map(|v| v * scale);
            Some(
                SimBox::new(Cell::Matrix(h), origin, sb.pbc())
                    .map_err(|e| invalid_data(format!("XTC box: {e:?}")))?,
            )
        }
        None => None,
    };
    frame.meta.insert("step", hdr.step);
    // XTC stores `time` and `precision` as `f32`; the store has one float
    // (`F = f64`), so they are widened on the way in.
    frame.meta.insert("time", hdr.time as f64);
    if precision != 0.0 {
        frame.meta.insert("precision", precision as f64);
    }
    // Every door reads through here, so every door checks here.
    crate::io::reader::check_read_frame(&frame)?;
    Ok(Some(frame))
}

fn parse_frame_at<R: BufRead + Seek>(r: &mut R, offset: u64) -> Result<Frame> {
    r.seek(SeekFrom::Start(offset))?;
    parse_frame_here(r)?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "XTC EOF at expected frame offset",
        )
    })
}

/// Scan the file, recording each frame's start byte offset.
fn scan_offsets<R: BufRead + Seek>(r: &mut R) -> Result<Vec<u64>> {
    let end = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let mut offsets = Vec::new();
    loop {
        let pos = r.stream_position()?;
        if pos >= end {
            break;
        }
        let hdr = match read_header(r) {
            Ok(h) => h,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                return Err(invalid_data(format!(
                    "XTC scan: header for frame {} at offset {pos} (end {end}): {e}",
                    offsets.len()
                )));
            }
        };
        let size = xdr::decode_i32(r)?;
        if size as usize != hdr.natoms {
            return Err(invalid_data("XTC coord size mismatch during scan"));
        }
        if hdr.natoms <= 9 {
            r.seek(SeekFrom::Current((hdr.natoms * DIM * 4) as i64))?;
        } else {
            // skip precision(4) + minint(12) + maxint(12) + smallidx(4) = 32
            r.seek(SeekFrom::Current(32))?;
            let nbytes = read_nbytes(r, hdr.wide_nbytes)?;
            r.seek(SeekFrom::Current(xdr::pad4(nbytes) as i64))?;
        }
        offsets.push(pos);
    }
    Ok(offsets)
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// XTC trajectory reader: true sequential stream *or* O(1) indexed random access.
///
/// XTC is the compressed GROMACS trajectory format: an XDR (big-endian) stream
/// of per-frame records carrying step, time, box, and **lossily compressed**
/// coordinates (nm). Compression quantizes each coordinate to an integer
/// (`round(x * precision)`), then bit-packs the integers, exploiting that
/// consecutive atoms are usually spatially close.
///
/// # Per-frame header (XDR, big-endian)
///
/// ```text
/// magic   i32 = 1995 (also accepts 2023, a forward-magic test variant)
/// natoms  i32
/// step    i32
/// time    f32
/// box     9 × f32 (3×3, nm)
/// ```
///
/// followed by the coordinate block:
///
/// ```text
/// size      i32 (== natoms)
/// if natoms <= 9:  3*natoms uncompressed f32
/// else:
///   precision f32
///   minint[3] i32   maxint[3] i32   smallidx i32
///   nbytes    i32
///   buf[nbytes] (XDR opaque, padded to 4)   -- the compressed bitstream
/// ```
///
/// The compression codec (`magicints` table, `receivebits`/`receiveints`
/// decode, `sendbits`/`sendints` encode) is a clean-room reimplementation of
/// the documented `xdr3dfcoord` algorithm — not transcribed from xdrfile or
/// any GPL source. Its unit tests round-trip hand-built frames through the
/// encoder and decoder.
///
/// # Output Frame
///
/// - `atoms` block: `id` (1-based), `x`/`y`/`z` (Å; the file's nm converted on
///   read and back on write).
/// - `frame.simbox`: from the box (row-stored vectors → column-stored H, in Å).
/// - `frame.meta`: `step`, `time`, `precision`.
pub struct XtcReader<R: BufRead + Seek> {
    reader: R,
    offsets: OnceLock<Vec<u64>>,
    cursor: usize,
}

impl<R: BufRead + Seek> XtcReader<R> {
    /// Wrap `reader`; index building is deferred.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            offsets: OnceLock::new(),
            cursor: 0,
        }
    }

    fn ensure_index(&mut self) -> Result<()> {
        if self.offsets.get().is_some() {
            return Ok(());
        }
        let offs = scan_offsets(&mut self.reader)?;
        self.offsets
            .set(offs)
            .map_err(|_| std::io::Error::other("failed to set XTC index"))?;
        Ok(())
    }

    /// Rewind for a fresh sequential pass. Does not clear an existing index.
    pub fn rewind(&mut self) -> Result<()> {
        self.cursor = 0;
        self.reader.seek(SeekFrom::Start(0))?;
        Ok(())
    }
}

impl<R: BufRead + Seek> Reader for XtcReader<R> {
    type R = R;
    fn new(reader: Self::R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for XtcReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        if let Some(offs) = self.offsets.get() {
            let off = match offs.get(self.cursor).copied() {
                Some(o) => o,
                None => return Ok(None),
            };
            let frame = parse_frame_at(&mut self.reader, off)?;
            self.cursor += 1;
            return Ok(Some(frame));
        }
        let frame = parse_frame_here(&mut self.reader)?;
        if frame.is_some() {
            self.cursor += 1;
        }
        Ok(frame)
    }
}

impl<R: BufRead + Seek> TrajectoryReader for XtcReader<R> {
    fn build_index(&mut self) -> Result<()> {
        self.ensure_index()
    }

    fn read_frame(&mut self, index: usize) -> Result<Option<Frame>> {
        self.ensure_index()?;
        let off = match self.offsets.get().and_then(|o| o.get(index).copied()) {
            Some(o) => o,
            None => return Ok(None),
        };
        self.cursor = index + 1;
        parse_frame_at(&mut self.reader, off).map(Some)
    }

    fn len(&mut self) -> Result<usize> {
        self.ensure_index()?;
        Ok(self.offsets.get().expect("index set").len())
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

const DEFAULT_PRECISION: f32 = 1000.0;

fn axis<FA: FrameAccess>(frame: &FA, key: &str) -> Option<Vec<f64>> {
    frame
        .column("atoms", key)
        .and_then(|c| c.get::<f64>())
        .map(|view| view.iter().copied().collect::<Vec<f64>>())
}

fn write_xtc_frame<W: Write, FA: FrameAccess>(w: &mut W, frame: &FA) -> Result<()> {
    // Every door writes through here, so every door checks here.
    crate::io::writer::check_write_frame(frame)?;
    crate::io::null_cells::refuse_null_cells(frame, "atoms", &["x", "y", "z"])?;
    let natoms = frame
        .visit_block("atoms", |a| a.n_rows().unwrap_or(0))
        .ok_or_else(|| invalid_data("XTC write: frame has no atoms block"))?;
    if natoms == 0 {
        return Err(invalid_data("XTC write: atoms block is empty"));
    }
    let xs = axis(frame, "x").ok_or_else(|| invalid_data("XTC write: atoms.x missing"))?;
    let ys = axis(frame, "y").ok_or_else(|| invalid_data("XTC write: atoms.y missing"))?;
    let zs = axis(frame, "z").ok_or_else(|| invalid_data("XTC write: atoms.z missing"))?;
    if xs.len() != natoms || ys.len() != natoms || zs.len() != natoms {
        return Err(invalid_data(
            "XTC write: coordinate columns disagree on length",
        ));
    }

    let meta = frame.meta_ref();
    let step = meta
        .get("step")
        .and_then(|value| value.as_i32())
        .unwrap_or(0);
    let time = meta
        .get("time")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0) as f32;
    let precision: f32 = meta
        .get("precision")
        .and_then(|value| value.as_f64())
        .filter(|&p| p > 0.0)
        .map(|p| p as f32)
        .unwrap_or(DEFAULT_PRECISION);

    xdr::encode_i32(w, XTC_MAGIC)?;
    xdr::encode_i32(w, natoms as i32)?;
    xdr::encode_i32(w, step)?;
    xdr::encode_f32(w, time)?;
    if let Some(sb) = frame.simbox_ref() {
        // Å → nm on the way out, mirroring the reader.
        let h = ndarray::arr2(&sb.matrix()) / NM_TO_ANGSTROM.get();
        for i in 0..DIM {
            for j in 0..DIM {
                xdr::encode_f32(w, h[(j, i)] as f32)?;
            }
        }
    } else {
        for _ in 0..9 {
            xdr::encode_f32(w, 0.0)?;
        }
    }

    xdr::encode_i32(w, natoms as i32)?;
    if natoms <= 9 {
        for a in 0..natoms {
            xdr::encode_f32(w, (xs[a] / NM_TO_ANGSTROM.get()) as f32)?;
            xdr::encode_f32(w, (ys[a] / NM_TO_ANGSTROM.get()) as f32)?;
            xdr::encode_f32(w, (zs[a] / NM_TO_ANGSTROM.get()) as f32)?;
        }
        return Ok(());
    }

    let mut coords = Vec::with_capacity(natoms * DIM);
    for a in 0..natoms {
        coords.push(xs[a] / NM_TO_ANGSTROM.get());
        coords.push(ys[a] / NM_TO_ANGSTROM.get());
        coords.push(zs[a] / NM_TO_ANGSTROM.get());
    }
    let (minint, maxint, smallidx, buf) = compress_coords(&coords, natoms, precision)?;
    xdr::encode_f32(w, precision)?;
    for v in minint {
        xdr::encode_i32(w, v)?;
    }
    for v in maxint {
        xdr::encode_i32(w, v)?;
    }
    xdr::encode_i32(w, smallidx)?;
    xdr::encode_i32(w, buf.len() as i32)?;
    xdr::encode_opaque(w, &buf)?;
    Ok(())
}

/// XTC trajectory writer.
pub struct XtcWriter<W: Write> {
    writer: W,
}

impl<W: Write> XtcWriter<W> {
    /// Create a new XTC writer.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> Writer for XtcWriter<W> {
    type W = W;
    fn new(writer: Self::W) -> Self {
        Self::new(writer)
    }
}

impl<W: Write> FrameWriter for XtcWriter<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        write_xtc_frame(&mut self.writer, frame)?;
        self.writer.flush()
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Read the first frame of an XTC file.
pub fn read_xtc<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    parse_frame_at(&mut src.into().reader()?, 0)
}

/// Write one frame as a one-frame XTC file (Å → nm).
pub fn write_xtc<'a>(sink: impl Into<Sink<'a>>, frame: &impl FrameAccess) -> Result<()> {
    sink.into().put_with(|w| write_xtc_frame(w, frame))
}

impl XtcReader<Box<dyn ReadSeek>> {
    /// Open an XTC file, or the bytes of one, for random access by frame.
    pub fn open<'a>(src: impl Into<Source<'a>>) -> Result<Self> {
        Ok(Self::new(src.into().owned_reader()?))
    }
}

impl<'a> XtcWriter<SinkWriter<'a>> {
    /// Create an XTC file, or fill a buffer, frame by frame.
    pub fn create(sink: impl Into<Sink<'a>>) -> Result<Self> {
        Ok(Self::new(sink.into().writer()?))
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Byte length of one complete XTC frame starting at `bytes[0]`.
///
/// `Ok(None)` means the prefix is valid so far but truncated.
fn try_xtc_frame_len(bytes: &[u8]) -> Result<Option<u32>> {
    if bytes.is_empty() {
        return Ok(None);
    }
    if bytes.len() < 4 {
        return Ok(None);
    }
    let magic = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
    if magic != XTC_MAGIC && magic != XTC_MAGIC_2023 {
        return Err(invalid_data(format!(
            "bad XTC magic {magic} (expected {XTC_MAGIC} or {XTC_MAGIC_2023})"
        )));
    }
    if bytes.len() < 52 {
        return Ok(None);
    }
    let natoms = i32::from_be_bytes(bytes[4..8].try_into().unwrap());
    if natoms <= 0 {
        return Err(invalid_data(format!("invalid_data XTC natoms {natoms}")));
    }
    let natoms = natoms as usize;
    let wide = magic == XTC_MAGIC_2023;
    let mut off = 52usize;
    if bytes.len() < off + 4 {
        return Ok(None);
    }
    let size = i32::from_be_bytes(bytes[off..off + 4].try_into().unwrap());
    off += 4;
    if size as usize != natoms {
        return Err(invalid_data(format!(
            "XTC coord size {size} disagrees with header natoms {natoms}"
        )));
    }
    if natoms <= 9 {
        let need = natoms * DIM * 4;
        if bytes.len() < off + need {
            return Ok(None);
        }
        return u32::try_from(off + need)
            .map(Some)
            .map_err(|_| invalid_data("XTC frame larger than 4 GiB"));
    }
    // precision(4) + minint(12) + maxint(12) + smallidx(4)
    if bytes.len() < off + 32 {
        return Ok(None);
    }
    off += 32;
    let (nbytes, nbytes_width) = if wide {
        if bytes.len() < off + 8 {
            return Ok(None);
        }
        let n = i64::from_be_bytes(bytes[off..off + 8].try_into().unwrap());
        (n, 8usize)
    } else {
        if bytes.len() < off + 4 {
            return Ok(None);
        }
        let n = i32::from_be_bytes(bytes[off..off + 4].try_into().unwrap()) as i64;
        (n, 4usize)
    };
    if nbytes < 0 {
        return Err(invalid_data(format!("negative XTC nbytes {nbytes}")));
    }
    off += nbytes_width;
    let padded = xdr::pad4(nbytes as usize);
    if bytes.len() < off + padded {
        return Ok(None);
    }
    u32::try_from(off + padded)
        .map(Some)
        .map_err(|_| invalid_data("XTC frame larger than 4 GiB"))
}

/// Streaming frame indexer for XTC files. Frames are self-describing;
/// the scanner measures each header + compressed block without decoding.
pub struct XtcIndexBuilder {
    scan: BinaryFrameScanner,
    error: Option<std::io::Error>,
}

impl Default for XtcIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl XtcIndexBuilder {
    /// Create an empty indexer.
    pub fn new() -> Self {
        Self {
            scan: BinaryFrameScanner::new(),
            error: None,
        }
    }
}

impl FrameIndexBuilder for XtcIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        if self.error.is_some() {
            return;
        }
        if let Err(err) = self.scan.feed(chunk, global_offset, try_xtc_frame_len) {
            self.error = Some(err);
        }
    }

    fn drain(&mut self) -> Vec<FrameOffset> {
        self.scan.drain()
    }

    fn finish(self: Box<Self>) -> Result<Vec<FrameOffset>> {
        if let Some(err) = self.error {
            return Err(err);
        }
        self.scan.finish(try_xtc_frame_len)
    }

    fn bytes_seen(&self) -> u64 {
        self.scan.bytes_seen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;

    #[test]
    fn sizeofint_known_values() {
        assert_eq!(sizeofint(1), 1);
        assert_eq!(sizeofint(2), 2);
        assert_eq!(sizeofint(255), 8);
        assert_eq!(sizeofint(256), 9);
    }

    #[test]
    fn sizeofints_matches_reference() {
        // The reference algorithm is byte-aligned-conservative: it accumulates
        // the mixed-radix product byte by byte, so 8*8*8 = 512 spans 2 bytes
        // whose high byte (2) needs 2 bits → 2 + 8 = 10 (not the theoretical 9).
        // Reproducing this exactly is what lets us read real GROMACS streams.
        assert_eq!(sizeofints(3, &[8, 8, 8]), 10);
        assert_eq!(sizeofints(3, &[1, 1, 1]), 1);
        assert_eq!(sizeofints(3, &[256, 256, 256]), 25);
    }

    #[test]
    fn bit_round_trip_single_value() {
        for &(nbits, val) in &[(1u32, 1u32), (5, 19), (8, 200), (13, 5000), (20, 1_000_000)] {
            let mut bw = BitWriter::new();
            bw.sendbits(nbits as i32, val);
            let buf = bw.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(br.receivebits(nbits as i32).unwrap(), val as i32);
        }
    }

    #[test]
    fn ints_round_trip() {
        let sizes = [901u32, 9001, 90001];
        let nbits = sizeofints(3, &sizes);
        let nums = [123u32, 4567, 89012];
        let mut bw = BitWriter::new();
        bw.sendints(3, nbits, &sizes, &nums);
        let buf = bw.finish();
        let mut br = BitReader::new(&buf);
        let mut got = [0i32; 3];
        br.receiveints(3, nbits, &sizes, &mut got).unwrap();
        assert_eq!(got, [123, 4567, 89012]);
    }

    #[test]
    fn compress_decompress_round_trip() {
        // A handful of atoms (>9 so the compressed path runs), nm coords.
        let coords: Vec<f64> = (0..30)
            .map(|k| (k as f64) * 0.037 - 0.5)
            .collect::<Vec<_>>();
        let natoms = coords.len() / 3;
        let precision = 1000.0f32;
        let (minint, maxint, smallidx, buf) = compress_coords(&coords, natoms, precision).unwrap();
        let back = decompress_coords(&buf, natoms, precision, minint, maxint, smallidx).unwrap();
        assert_eq!(back.len(), coords.len());
        for (a, b) in coords.iter().zip(back.iter()) {
            assert!((a - b).abs() <= 1.0 / precision as f64 + 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn build_simbox_zero_is_none() {
        assert!(build_simbox(&[0.0; 9]).is_none());
        assert!(build_simbox(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]).is_some());
    }

    fn xtc_frame(natoms: usize, x0: f64) -> Frame {
        let mut atoms = Block::new();
        let ids: Vec<Idx> = (1..=natoms as Idx).collect();
        let mut xs = vec![0.0; natoms];
        let ys = vec![0.0; natoms];
        let zs = vec![0.0; natoms];
        xs[0] = x0;
        atoms
            .insert("id", Array1::from(ids).into_dyn(), None)
            .unwrap();
        atoms
            .insert("x", Array1::from(xs).into_dyn(), None)
            .unwrap();
        atoms
            .insert("y", Array1::from(ys).into_dyn(), None)
            .unwrap();
        atoms
            .insert("z", Array1::from(zs).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        let h = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]];
        frame.simbox = Some(SimBox::new(Cell::Matrix(h), [0.0, 0.0, 0.0], [true; 3]).unwrap());
        frame
    }

    fn write_xtc_mem(frames: &[Frame]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut writer = XtcWriter::new(&mut buf);
            for frame in frames {
                FrameWriter::write(&mut writer, frame).expect("write xtc");
            }
        }
        buf
    }

    fn xtc_index_chunked(bytes: &[u8], chunk: usize) -> Vec<FrameOffset> {
        let mut builder = Box::new(XtcIndexBuilder::new());
        let mut offset = 0u64;
        let mut out = Vec::new();
        for piece in bytes.chunks(chunk.max(1)) {
            builder.feed(piece, offset);
            offset += piece.len() as u64;
            out.extend(builder.drain());
        }
        out.extend(builder.finish().expect("finish xtc index"));
        out
    }

    #[test]
    fn xtc_streaming_small_and_compressed_round_trip() {
        for natoms in [2usize, 12] {
            let frames = [xtc_frame(natoms, 1.5), xtc_frame(natoms, 2.5)];
            let bytes = write_xtc_mem(&frames);
            let one_shot = xtc_index_chunked(&bytes, bytes.len());
            assert_eq!(one_shot.len(), 2, "natoms={natoms}");
            for cs in [1usize, 9, 17, 64] {
                assert_eq!(
                    xtc_index_chunked(&bytes, cs),
                    one_shot,
                    "natoms={natoms} chunk={cs}"
                );
            }
            for (i, entry) in one_shot.iter().enumerate() {
                let lo = entry.byte_offset as usize;
                let hi = lo + entry.byte_len as usize;
                let parsed = read_xtc(&bytes[lo..hi]).expect("parse xtc");
                assert_eq!(parsed.get("atoms").unwrap().n_rows().unwrap(), natoms);
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
                        < 0.02,
                    "xtc coord drift"
                );
            }
        }
    }
}
