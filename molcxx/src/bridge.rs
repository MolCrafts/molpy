use super::*;

#[cxx::bridge(namespace = "molrs")]
pub mod ffi {
    /// Chemical element exported from molrs' canonical Rust periodic table.
    ///
    /// The variants are injected by build.rs from
    /// `crates/molrs/src/core/element.rs`; this bridge never owns a second
    /// hand-maintained element table.
    #[repr(u8)]
    enum Element {
        H = 1,
        He = 2,
        Li = 3,
        Be = 4,
        B = 5,
        C = 6,
        N = 7,
        O = 8,
        F = 9,
        Ne = 10,
        Na = 11,
        Mg = 12,
        Al = 13,
        Si = 14,
        P = 15,
        S = 16,
        Cl = 17,
        Ar = 18,
        K = 19,
        Ca = 20,
        Sc = 21,
        Ti = 22,
        V = 23,
        Cr = 24,
        Mn = 25,
        Fe = 26,
        Co = 27,
        Ni = 28,
        Cu = 29,
        Zn = 30,
        Ga = 31,
        Ge = 32,
        As = 33,
        Se = 34,
        Br = 35,
        Kr = 36,
        Rb = 37,
        Sr = 38,
        Y = 39,
        Zr = 40,
        Nb = 41,
        Mo = 42,
        Tc = 43,
        Ru = 44,
        Rh = 45,
        Pd = 46,
        Ag = 47,
        Cd = 48,
        In = 49,
        Sn = 50,
        Sb = 51,
        Te = 52,
        I = 53,
        Xe = 54,
        Cs = 55,
        Ba = 56,
        La = 57,
        Ce = 58,
        Pr = 59,
        Nd = 60,
        Pm = 61,
        Sm = 62,
        Eu = 63,
        Gd = 64,
        Tb = 65,
        Dy = 66,
        Ho = 67,
        Er = 68,
        Tm = 69,
        Yb = 70,
        Lu = 71,
        Hf = 72,
        Ta = 73,
        W = 74,
        Re = 75,
        Os = 76,
        Ir = 77,
        Pt = 78,
        Au = 79,
        Hg = 80,
        Tl = 81,
        Pb = 82,
        Bi = 83,
        Po = 84,
        At = 85,
        Rn = 86,
        Fr = 87,
        Ra = 88,
        Ac = 89,
        Th = 90,
        Pa = 91,
        U = 92,
        Np = 93,
        Pu = 94,
        Am = 95,
        Cm = 96,
        Bk = 97,
        Cf = 98,
        Es = 99,
        Fm = 100,
        Md = 101,
        No = 102,
        Lr = 103,
        Rf = 104,
        Db = 105,
        Sg = 106,
        Bh = 107,
        Hs = 108,
        Mt = 109,
        Ds = 110,
        Rg = 111,
        Cn = 112,
        Nh = 113,
        Fl = 114,
        Mc = 115,
        Lv = 116,
        Ts = 117,
        Og = 118,
    }

    /// Exact frame-metadata dtype.
    #[repr(u8)]
    enum MetaType {
        Bool,
        I32,
        I64,
        U32,
        U64,
        F64,
        String,
        Bool3,
        I32x3,
        I64x3,
        U32x3,
        U64x3,
        F64x3,
        F64x6,
        F64x9,
    }

    /// One frame-metadata value with its key and exact dtype. Only the payload
    /// selected by `dtype` is used.
    struct KeyedMetaValue {
        key: String,
        dtype: MetaType,
        bool_value: bool,
        i32_value: i32,
        i64_value: i64,
        u32_value: u32,
        u64_value: u64,
        f64_value: f64,
        string_value: String,
        bool_values: Vec<u8>,
        i32_values: Vec<i32>,
        i64_values: Vec<i64>,
        u32_values: Vec<u32>,
        u64_values: Vec<u64>,
        f64_values: Vec<f64>,
    }

    /// A line fitted through one channel of a `TimeCorrelation` (molrs
    /// `compute::LinearFitResult`); `fit_start..=fit_end` are lag indices.
    struct LinearFitResult {
        slope: f64,
        intercept: f64,
        r2: f64,
        fit_start: u64,
        fit_end: u64,
    }

    /// g(r) over one or more frames (molrs `compute::RdfResult`): bin edges and
    /// centers in Å, the normalized curve and the raw counts.
    struct RdfResult {
        bin_edges: Vec<f64>,
        bin_centers: Vec<f64>,
        rdf: Vec<f64>,
        n_r: Vec<f64>,
        n_points: u64,
        n_query_points: u64,
        self_query: bool,
        volume: f64,
        r_min: f64,
        n_frames: u64,
        dimensionality: u8,
    }

    extern "Rust" {
        // ── Exact consumer contract ───────────────────────────────
        // Capabilities let consumers fail loudly when a required surface was
        // compiled out or omitted.
        fn cxx_api_capabilities() -> u64;

        // ── Frame bridge (molrs.core.Frame via molrs-ffi FrameRef) ─────
        type FrameRef;

        // ── Region bridge (molrs.core.Region via molrs-ffi RegionRef) ──
        // A region answers a signed distance; `contains` is its sign and
        // `bounds` the box it fits in. Compositions are ordinary handles, so
        // a shell is `region_and(outer, region_not(inner))`.
        type RegionRef;

        // Every vector argument is exactly 3 values; anything else is an error.
        fn region_sphere(center: &[f64], radius: f64) -> Result<Box<RegionRef>>;
        fn region_cuboid(origin: &[f64], lengths: &[f64]) -> Result<Box<RegionRef>>;
        fn region_half_space(normal: &[f64], point: &[f64]) -> Result<Box<RegionRef>>;
        fn region_cylinder(
            base: &[f64],
            axis: &[f64],
            radius: f64,
            length: f64,
        ) -> Result<Box<RegionRef>>;
        fn region_ellipsoid(center: &[f64], semi_axes: &[f64]) -> Result<Box<RegionRef>>;

        fn region_and(a: &RegionRef, b: &RegionRef) -> Box<RegionRef>;
        fn region_or(a: &RegionRef, b: &RegionRef) -> Box<RegionRef>;
        fn region_not(a: &RegionRef) -> Box<RegionRef>;

        // `points` is flat [x, y, z, ...]; a ragged length is an error.
        fn region_distance(rref: &RegionRef, points: &[f64]) -> Result<Vec<f64>>;
        fn region_contains(rref: &RegionRef, points: &[f64]) -> Result<Vec<u8>>;
        fn region_bounds(rref: &RegionRef) -> Vec<f64>;

        fn frame_new() -> Box<FrameRef>;

        // Cross-extension ingress: rebuild a bridge handle from the raw
        // address of a molrs-python `*mut molrs_ffi::FrameRef` (carried by
        // `Frame._ffi_frameref_capsule()`). `unsafe` — caller must pass a
        // live pointer; see the `# Safety` note on the Rust impl.
        unsafe fn frame_clone_from_addr(addr: usize) -> Box<FrameRef>;

        // introspection
        fn frame_block_names(fref: &FrameRef) -> Vec<String>;
        fn frame_has_block(fref: &FrameRef, block: &str) -> bool;
        fn frame_block_columns(fref: &FrameRef, block: &str) -> Vec<String>;
        fn frame_block_n_rows(fref: &FrameRef, block: &str) -> i64;

        // metadata: keys in insertion order, one value by key, insert/replace
        fn frame_meta_keys(fref: &FrameRef) -> Vec<String>;
        fn frame_get_meta(fref: &FrameRef, key: &str) -> Result<KeyedMetaValue>;
        fn frame_set_meta(fref: &mut FrameRef, value: KeyedMetaValue) -> Result<()>;

        // readers — owned copies (RefCell precludes returning borrowed slices);
        // an absent block or column reads as empty, a column of another dtype
        // is an error naming both dtypes. The box is its cell matrix H, 9
        // row-major values (empty when the frame has no box or the box has no
        // defined cell).
        //
        // One reader and one writer per column dtype is deliberate: cxx has no
        // generics, so `frame_column::<T>` cannot cross the bridge — the
        // `_{f64,i32,u64,str}` suffix is the type parameter spelled out, not a
        // variant family.
        fn frame_column_f64(fref: &FrameRef, block: &str, col: &str) -> Result<Vec<f64>>;
        fn frame_column_i32(fref: &FrameRef, block: &str, col: &str) -> Result<Vec<i32>>;
        fn frame_column_u64(fref: &FrameRef, block: &str, col: &str) -> Result<Vec<u64>>;
        fn frame_column_str(fref: &FrameRef, block: &str, col: &str) -> Result<Vec<String>>;
        fn frame_box_h(fref: &FrameRef) -> Vec<f64>;

        // create-or-update writers (one per dtype, as the readers)
        fn frame_set_column_f64(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[f64],
        ) -> Result<()>;
        fn frame_set_column_i32(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[i32],
        ) -> Result<()>;
        fn frame_set_column_u64(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[u64],
        ) -> Result<()>;
        fn frame_set_column_str(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            data: &[String],
        ) -> Result<()>;
        // a periodic box at the origin from its 9-value row-major H
        fn frame_set_box_h(fref: &mut FrameRef, h: &[f64]) -> Result<()>;
        // explicit species conversion (molrs Element::fill_atomic_numbers /
        // Element::fill_symbols): atoms.element symbols -> atoms.atomic_number
        // (Uint; "*" is 0), and back (0 is "*"). Nothing else converts; an
        // unknown symbol or a number outside 0..=118 is an error naming the row.
        fn frame_fill_atomic_numbers(fref: &mut FrameRef) -> Result<()>;
        fn frame_fill_symbols(fref: &mut FrameRef) -> Result<()>;
        // declared precision of an f64 column: a `*.mrec` writer stores it
        // rounded to the largest power of two <= precision (molrec)
        fn frame_set_precision(
            fref: &mut FrameRef,
            block: &str,
            col: &str,
            precision: f64,
        ) -> Result<()>;

        // ── AM1-BCC bridge (molrs::ff::charge::BccModel) ─────────
        // Atomiverse supplies AM1 base charges; molrs owns BCC typing. The
        // frame carries atoms.{atomic_number, x, y, z} + bonds.{atomi, atomj};
        // species is read from atomic_number only (no `element` needed).
        //
        // `bcc_model_new` selects the correction family by name — "bcc"
        // (BCCPARM.DAT, model id `"bcc"`) or `"abcg2"` (BCCPARM_ABCG2.DAT,
        // `-c abcg2`). A name molrs does not know is refused, never defaulted.
        // `assign` is the whole model (equivalence, then correct); `correct`
        // is the correction stage alone, for equivalenced base charges. Both
        // write `atoms.charge` and return the charges.
        //
        // Every call returns `Result`, and that is load-bearing: cxx marks every
        // non-Result `extern "Rust"` fn `noexcept`, so a Rust panic could only
        // abort the calling process. The errors these can raise are the caller's
        // CHEMISTRY — a molecule with no BCC correction row (boron), a missing
        // atom type, a missing bond order — not programmer bugs, so they cross
        // as a catchable `rust::Error` and leave the engine alive to handle them.
        type BccModel;
        fn bcc_model_new(parameter_set: &str) -> Result<Box<BccModel>>;
        fn assign(self: &BccModel, fref: &mut FrameRef, am1_charges: &[f64]) -> Result<Vec<f64>>;
        fn correct(self: &BccModel, fref: &mut FrameRef, am1_charges: &[f64]) -> Result<Vec<f64>>;

        // ── I/O ──────────────────────────────────────────────────
        // One frame currency: every writer takes a `FrameRef` (built with
        // `frame_new` + `frame_set_column_*` + `frame_set_box_h` +
        // `frame_set_meta`) and every reader returns one. Species is the
        // schema's Uint `atomic_number` (0..=118, 0 = wildcard): every writer,
        // the streaming one included, refuses a number outside that range,
        // naming the row. The caller's frame is never modified by a writer.

        // Write one frame to an XYZ / ExtXYZ file (molrs XyzWriter): its atoms
        // columns, box and typed metadata. XYZ spells symbols, so `element` is
        // filled from `atoms.atomic_number` explicitly (Element::fill_symbols)
        // on a copy; a missing atomic_number or a number outside 0..=118 is an
        // error naming the row. append=false truncates; append=true appends.
        fn write_xyz(path: &str, fref: &FrameRef, append: bool) -> Result<()>;
        // Read the first frame of an (ext)XYZ file into a materialize-ready
        // FrameRef (atoms.{x,y,z,atomic_number} + box). `atomic_number` is
        // converted explicitly (Element::fill_atomic_numbers) from the
        // required ExtXYZ species column, which is then dropped: the engine's
        // boundary schema carries Z only. All XYZ parsing lives in molrs
        // (io::read_xyz).
        fn read_xyz(path: &str) -> Result<Box<FrameRef>>;

        // Write one frame as a `*.mrec` record whose `frame` section is that
        // frame — every block, column, declared precision, box and metadata
        // value (molrs io::write_mrec); read it back with read_mrec_frame
        // (molrs MrecFile::frame). mrec is molrs' native format: it stores
        // `atomic_number` itself, no `element` symbols.
        fn write_mrec_frame(path: &str, fref: &FrameRef) -> Result<()>;
        fn read_mrec_frame(path: &str) -> Result<Box<FrameRef>>;
        // Frame `index` of the trajectory in a `*.mrec` record (molrs
        // MrecReader::frame); reads what an MrecWriterRef wrote.
        fn read_mrec_trajectory_frame(path: &str, index: u64) -> Result<Box<FrameRef>>;

        // ── Streaming `*.mrec` trajectory writer (molrs MrecWriter) ──
        // The engine's output path: one writer per run, one frame per
        // append; complete inner chunks land on their own, `flush` commits
        // (durably when `durable`), `close` ends the run. `flush_every == 0`
        // leaves the landing cadence to the writer. `schema_from` pins the
        // blocks/columns every later frame must stay inside.
        type MrecWriterRef;
        fn mrec_writer_create(
            path: &str,
            schema_from: &FrameRef,
            flush_every: u64,
            durable: bool,
        ) -> Result<Box<MrecWriterRef>>;
        fn mrec_writer_open(
            path: &str,
            flush_every: u64,
            durable: bool,
        ) -> Result<Box<MrecWriterRef>>;
        fn mrec_writer_append(
            writer: &mut MrecWriterRef,
            fref: &FrameRef,
            step: i64,
            time: f64,
            has_time: bool,
        ) -> Result<()>;
        fn mrec_writer_flush(writer: &mut MrecWriterRef) -> Result<()>;
        fn mrec_writer_committed(writer: &MrecWriterRef) -> u64;
        fn mrec_writer_close(writer: Box<MrecWriterRef>) -> Result<()>;

        // ── Trajectory analysis (mirrors molrs::compute) ─────────
        // Each type is the molrs analysis of the same name. Its constructor
        // (`*_new`, plus `set_*` for an optional choice) takes configuration
        // only; data goes to `push` on the streaming accumulator that
        // `accumulator()` hands out — one frame per call, bounded memory —
        // and `finish` reads the result. There is no whole-trajectory
        // `compute`: a stored trajectory is pushed frame by frame. Every
        // fallible call returns `Result`: a bad argument, a ragged buffer or a
        // molrs compute error throws `rust::Error` — never an empty curve, a
        // NaN or `false`.
        //
        // Buffers: a frame is blocked by component (positions x[0..n) y[..)
        // z[..); velocities any n_dof values); a box is a 9-value row-major
        // cell matrix H. An optional box argument is empty for "no box".

        // A correlation curve (molrs TimeCorrelation): `lag_times` in the unit
        // of dt, one curve per channel (`channel` throws for a missing one).
        type TimeCorrelation;
        fn lag_times(self: &TimeCorrelation) -> Vec<f64>;
        fn n_lags(self: &TimeCorrelation) -> u64;
        fn n_channels(self: &TimeCorrelation) -> u64;
        fn channel(self: &TimeCorrelation, channel: u64) -> Result<Vec<f64>>;

        // The streaming accumulator of Msd / Vacf (molrs CorrelatorAccumulator).
        // `push` takes one frame, the sample interval dt (the same every frame)
        // and, for Msd, the frame's H to reject a wrapped trajectory (empty to
        // skip); `finish` reads the curve so far without consuming the state.
        type CorrelatorAccumulator;
        fn push(self: &mut CorrelatorAccumulator, frame: &[f64], dt: f64, h: &[f64]) -> Result<()>;
        fn n_frames(self: &CorrelatorAccumulator) -> u64;
        fn finish(self: &CorrelatorAccumulator) -> Result<Box<TimeCorrelation>>;

        // Mean squared displacement over lags 0..=max_lag frames of unwrapped
        // positions. Origins default to "all" (windowed estimator); "first"
        // measures from frame 0. The Einstein D is a LinearFit of the curve:
        // D = slope / (2 * dims).
        type Msd;
        fn msd_new(max_lag: u64) -> Box<Msd>;
        fn set_origins(self: &mut Msd, origins: &str) -> Result<()>;
        // Every `stride`-th frame is a time origin (origins "all"); 0 is an error.
        fn set_stride(self: &mut Msd, stride: u64) -> Result<()>;
        fn accumulator(self: &Msd) -> Box<CorrelatorAccumulator>;

        // Velocity autocorrelation over lags 0..=max_lag frames, averaged over
        // degrees of freedom.
        type Vacf;
        fn vacf_new(max_lag: u64) -> Box<Vacf>;
        // "unbiased" (default), "biased" or "none"; every `stride`-th origin;
        // "subtract" (default) or "keep" each component's time mean.
        fn set_normalization(self: &mut Vacf, normalization: &str) -> Result<()>;
        fn set_stride(self: &mut Vacf, stride: u64) -> Result<()>;
        fn set_mean(self: &mut Vacf, mean: &str) -> Result<()>;
        fn accumulator(self: &Vacf) -> Box<CorrelatorAccumulator>;

        // Ordinary-least-squares line through one channel of a curve, over
        // the window (start, end) given as fractions of the last lag.
        type LinearFit;
        fn linear_fit_new(start: f64, end: f64) -> Result<Box<LinearFit>>;
        fn compute(
            self: &LinearFit,
            correlation: &TimeCorrelation,
            channel: u64,
        ) -> Result<LinearFitResult>;

        // Radial distribution function g(r) over [r_min, r_max) Å in n_bins
        // bins (note r_max before r_min). Each frame's pairs come from a
        // periodic cell-list self-query with cutoff r_max; the accumulator
        // takes one frame and its own H per `push` (NPT supported).
        type Rdf;
        fn rdf_new(n_bins: u64, r_max: f64, r_min: f64) -> Result<Box<Rdf>>;
        fn accumulator(self: &Rdf) -> Box<RdfAccumulator>;

        type RdfAccumulator;
        fn push(self: &mut RdfAccumulator, positions: &[f64], h: &[f64]) -> Result<()>;
        fn n_frames(self: &RdfAccumulator) -> u64;
        fn finish(self: &RdfAccumulator) -> Result<RdfResult>;
    }
}
