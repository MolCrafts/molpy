//! Live Frame streaming over WebSocket — Python view of `molrs::stream`.
//!
//! A producer (an MD loop in Python, Rust, or anything that can build a
//! [`Frame`]) binds a [`PyPublisher`] (or dials a relay with one) configured
//! by a [`PyPublisherConfig`], and calls `send()` once per step. The
//! call never blocks on network I/O: frames go through a bounded buffer that
//! drops the oldest payload when a client cannot keep up, so a slow viewer
//! slows nothing down. Viewers dial the socket and read the payloads back with
//! `molpy.io.read_wire_frame` ([`crate::io`]).
//!
//! Traffic in the other direction is [`PyControlCommand`]: a viewer asks the
//! producer to pause, change rate, or restrict the atom subset. The producer
//! decides what to do about it — nothing here interprets a command.
//!
//! # Platform
//!
//! [`PyControlCommand`] is portable; [`PyPublisher`] and [`PyPublisherConfig`]
//! open TCP sockets and are therefore native-only, gated exactly like
//! `molrs::stream::publisher`. A Pyodide build has the command type and no
//! server.
//!
//! [`Frame`]: molrs::core::Frame

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use molrs::stream::ControlCommand;

use crate::error::py_value_err;

/// Resolve a wire-encoding name onto [`FrameEncoding`].
///
/// The two spellings are the only ones the Rust side can produce, so an
/// unknown name is an error rather than a silent fall back to MessagePack —
/// a caller who writes `"messagepack"` must find out, not stream bytes the
/// peer will read as JSON.
pub(crate) fn frame_encoding(name: &str) -> PyResult<molrs::stream::FrameEncoding> {
    match name {
        "msgpack" => Ok(molrs::stream::FrameEncoding::MessagePack),
        "json" => Ok(molrs::stream::FrameEncoding::Json),
        other => Err(PyValueError::new_err(format!(
            "unknown wire format {other:?}; expected 'msgpack' or 'json'"
        ))),
    }
}

/// A control message sent from a streaming viewer back to the producer.
///
/// Immutable. Build one with the named constructors
/// (:meth:`pause`, :meth:`resume`, :meth:`set_frame_rate`,
/// :meth:`set_subset`, :meth:`request_key_frame`) and read it back through
/// :attr:`kind` plus whichever payload attribute that kind carries.
///
/// The wire form is owned by molrs, not by the caller: encode with
/// :meth:`to_bytes` and decode with :meth:`from_bytes` rather than hand-writing
/// ``{"type": "pause"}``.
// `skip_from_py_object`: a command is only ever handed *out* (by
// `Publisher.recv_command`) or built by a named constructor. No binding takes
// one as a parameter, so there is nothing to extract it for.
#[pyclass(
    module = "molpy.stream",
    name = "ControlCommand",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyControlCommand {
    pub(crate) inner: ControlCommand,
}

#[pymethods]
impl PyControlCommand {
    /// Ask the producer to stop advancing its simulation.
    #[staticmethod]
    fn pause() -> Self {
        Self {
            inner: ControlCommand::Pause,
        }
    }

    /// Ask the producer to resume after a :meth:`pause`.
    #[staticmethod]
    fn resume() -> Self {
        Self {
            inner: ControlCommand::Resume,
        }
    }

    /// Ask the producer to cap its stream rate.
    ///
    /// Parameters
    /// ----------
    /// hz : float
    ///     Desired frames per second. Must be finite and positive.
    #[staticmethod]
    fn set_frame_rate(hz: f64) -> PyResult<Self> {
        if !hz.is_finite() || hz <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "frame rate must be finite and positive, got {hz}"
            )));
        }
        Ok(Self {
            inner: ControlCommand::SetFrameRate { hz },
        })
    }

    /// Ask the producer to restrict later frames to `atom_ids`.
    ///
    /// Order is preserved on the wire — the producer sees the list it was sent.
    #[staticmethod]
    fn set_subset(atom_ids: Vec<u32>) -> Self {
        Self {
            inner: ControlCommand::SetSubset { atom_ids },
        }
    }

    /// Ask the producer to emit a full frame on its next send.
    #[staticmethod]
    fn request_key_frame() -> Self {
        Self {
            inner: ControlCommand::RequestKeyFrame,
        }
    }

    /// Wire tag naming this command: one of ``"pause"``, ``"resume"``,
    /// ``"set_frame_rate"``, ``"set_subset"``, ``"request_key_frame"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            ControlCommand::Pause => "pause",
            ControlCommand::Resume => "resume",
            ControlCommand::SetFrameRate { .. } => "set_frame_rate",
            ControlCommand::SetSubset { .. } => "set_subset",
            ControlCommand::RequestKeyFrame => "request_key_frame",
        }
    }

    /// Requested frames per second, or ``None`` for other kinds.
    #[getter]
    fn hz(&self) -> Option<f64> {
        match self.inner {
            ControlCommand::SetFrameRate { hz } => Some(hz),
            _ => None,
        }
    }

    /// Requested atom subset, or ``None`` for other kinds.
    #[getter]
    fn atom_ids(&self) -> Option<Vec<u32>> {
        match &self.inner {
            ControlCommand::SetSubset { atom_ids } => Some(atom_ids.clone()),
            _ => None,
        }
    }

    /// Encode this command for transmission to a producer.
    ///
    /// Parameters
    /// ----------
    /// format : {"json", "msgpack"}
    ///     Wire encoding. JSON is the default here — a control message is one
    ///     small object, and a browser console can produce it by hand.
    #[pyo3(signature = (format = "json"))]
    fn to_bytes<'py>(&self, py: Python<'py>, format: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .inner
            .to_bytes(frame_encoding(format)?)
            .map_err(py_value_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Decode a command a viewer sent.
    #[staticmethod]
    #[pyo3(signature = (data, format = "json"))]
    fn from_bytes(data: &[u8], format: &str) -> PyResult<Self> {
        let inner =
            ControlCommand::from_bytes(data, frame_encoding(format)?).map_err(py_value_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ControlCommand::SetFrameRate { hz } => {
                format!("ControlCommand(kind='set_frame_rate', hz={hz})")
            }
            ControlCommand::SetSubset { atom_ids } => format!(
                "ControlCommand(kind='set_subset', atom_ids=[{} ids])",
                atom_ids.len()
            ),
            _ => format!("ControlCommand(kind='{}')", self.kind()),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use server::{PyPublisher, PyPublisherConfig};

#[cfg(not(target_arch = "wasm32"))]
mod server {
    use std::time::Duration;

    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use molrs::stream::{Publisher, PublisherConfig};

    use super::PyControlCommand;
    use super::frame_encoding;
    use crate::core::frame::PyFrame;
    use crate::error::{io_error_to_pyerr, py_value_err};

    /// How a :class:`Publisher` streams: ``molrs::stream::PublisherConfig``.
    ///
    /// Parameters
    /// ----------
    /// format : {"msgpack", "json"}
    ///     Wire encoding for outbound frames. Viewers must decode with the
    ///     same format.
    /// buffer_size : int
    ///     How many encoded frames may be in flight before the oldest is
    ///     dropped. Must be at least 1.
    /// token : str or None
    ///     Shared secret clients must present in a ``hello`` handshake before
    ///     they receive frames. ``None`` accepts every connection — fine for
    ///     loopback, wrong for a shared bind.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     An unknown ``format`` or a ``buffer_size`` of 0.
    #[pyclass(
        module = "molpy.stream",
        name = "PublisherConfig",
        frozen,
        from_py_object
    )]
    #[derive(Clone)]
    pub struct PyPublisherConfig {
        format: String,
        buffer_size: usize,
        token: Option<String>,
    }

    #[pymethods]
    impl PyPublisherConfig {
        #[new]
        #[pyo3(signature = (*, format = "msgpack", buffer_size = 4, token = None))]
        fn new(format: &str, buffer_size: usize, token: Option<String>) -> PyResult<Self> {
            frame_encoding(format)?;
            if buffer_size == 0 {
                return Err(PyValueError::new_err(
                    "buffer_size must be at least 1; 0 would drop every frame",
                ));
            }
            Ok(Self {
                format: format.to_string(),
                buffer_size,
                token,
            })
        }

        /// The wire encoding, ``"msgpack"`` or ``"json"``.
        #[getter]
        fn format(&self) -> &str {
            &self.format
        }

        /// Encoded frames that may be in flight before the oldest is dropped.
        #[getter]
        fn buffer_size(&self) -> usize {
            self.buffer_size
        }

        /// The shared secret clients must present, or ``None``.
        #[getter]
        fn token(&self) -> Option<String> {
            self.token.clone()
        }

        fn __repr__(&self) -> String {
            format!(
                "PublisherConfig(format={:?}, buffer_size={}, token={})",
                self.format,
                self.buffer_size,
                if self.token.is_some() {
                    "'***'"
                } else {
                    "None"
                }
            )
        }
    }

    impl PyPublisherConfig {
        fn to_core(&self) -> PyResult<PublisherConfig> {
            Ok(PublisherConfig {
                format: frame_encoding(&self.format)?,
                buffer_size: self.buffer_size,
                max_frame_rate: 0.0,
                token: self.token.clone(),
            })
        }
    }

    /// WebSocket publisher that streams frames to viewers.
    ///
    /// Built by one of two doors, as in Rust: :meth:`bind` listens for viewers
    /// (a server), :meth:`connect` dials a relay. ``send`` is safe to call
    /// from inside a simulation loop: it encodes the frame, hands it to a
    /// background thread, and returns. When the buffer is full the oldest
    /// pending frame is dropped rather than stalling the producer, so the
    /// stream degrades to "latest frames only" instead of throttling the
    /// science.
    ///
    /// Use it as a context manager, or call :meth:`close` — either shuts the
    /// connection down and joins the background thread.
    ///
    /// Examples
    /// --------
    /// >>> config = molpy.stream.PublisherConfig()  # doctest: +SKIP
    /// >>> with molpy.stream.Publisher.bind("127.0.0.1:8765", config) as server:  # doctest: +SKIP
    /// ...     for _ in range(steps):
    /// ...         integrator.step()
    /// ...         server.send(integrator.frame)
    //
    // Deliberately not `unsendable`: `Publisher` is `Arc`-shared and
    // `Send + Sync`, and a producer commonly runs its loop on a worker thread.
    // (The `Frame` handed to `send` is still thread-bound — that is `PyFrame`'s
    // constraint, and it holds because each thread builds its own.)
    #[pyclass(module = "molpy.stream", name = "Publisher")]
    pub struct PyPublisher {
        inner: Option<Publisher>,
        address: String,
    }

    #[pymethods]
    impl PyPublisher {
        /// Bind a WebSocket listener viewers dial.
        ///
        /// Parameters
        /// ----------
        /// address : str
        ///     ``"host:port"``. Port ``0`` asks the OS for a free one; read
        ///     :attr:`address` afterwards to learn which.
        /// config : PublisherConfig
        ///     Wire format, buffer size and token.
        ///
        /// Raises
        /// ------
        /// OSError
        ///     The address cannot be bound.
        #[staticmethod]
        fn bind(address: &str, config: &PyPublisherConfig) -> PyResult<Self> {
            let server = Publisher::bind(address, config.to_core()?).map_err(io_error_to_pyerr)?;
            // A bound server always has one; `local_addr` is Option only
            // because a dialed publisher has no address to hand out.
            let bound = server
                .local_addr()
                .map(|a| a.to_string())
                .unwrap_or_else(|| address.to_string());
            Ok(Self {
                inner: Some(server),
                address: bound,
            })
        }

        /// Dial a relay at ``url`` (``"ws://host:port/path"``) and stream to it.
        ///
        /// Parameters
        /// ----------
        /// url : str
        ///     The relay's WebSocket URL.
        /// config : PublisherConfig
        ///     Wire format, buffer size and token.
        ///
        /// Raises
        /// ------
        /// OSError
        ///     The URL cannot be dialed.
        #[staticmethod]
        fn connect(url: &str, config: &PyPublisherConfig) -> PyResult<Self> {
            let server = Publisher::connect(url, config.to_core()?).map_err(io_error_to_pyerr)?;
            Ok(Self {
                inner: Some(server),
                address: url.to_string(),
            })
        }

        /// The bound ``"host:port"`` (resolved after an ephemeral-port bind),
        /// or the URL a connected publisher dialed.
        #[getter]
        fn address(&self) -> &str {
            &self.address
        }

        /// Number of viewers currently connected.
        #[getter]
        fn n_clients(&self) -> PyResult<usize> {
            Ok(self.server()?.n_clients())
        }

        /// Broadcast one frame. Returns immediately; never blocks on the network.
        fn send(&self, frame: &PyFrame) -> PyResult<()> {
            let server = self.server()?;
            // Encode straight out of the shared FFI store — no deep copy per
            // step. The GIL is deliberately held: that store is documented
            // single-threaded and GIL-guarded, so the borrow must not outlive
            // it. The work here is an encode plus a channel push, not network
            // I/O, which the background thread owns.
            frame.with_frame(|f| server.send(f))?.map_err(py_value_err)
        }

        /// Wait for the next control command from any viewer.
        ///
        /// Parameters
        /// ----------
        /// timeout : float
        ///     Seconds to wait. ``0`` polls and returns immediately.
        ///
        /// Returns
        /// -------
        /// ControlCommand or None
        ///     ``None`` when the timeout elapsed with no command pending.
        #[pyo3(signature = (timeout = 0.0))]
        fn recv_command(&self, py: Python<'_>, timeout: f64) -> PyResult<Option<PyControlCommand>> {
            if !timeout.is_finite() || timeout < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "timeout must be finite and non-negative, got {timeout}"
                )));
            }
            // Clone the handle (shared `Arc` state, not a second server) so the
            // closure captures only `Send` values and the GIL can be released
            // for the wait — otherwise a one-second poll freezes every other
            // Python thread.
            let server = self.server()?.clone();
            let duration = Duration::from_secs_f64(timeout);
            let received = py.detach(move || server.recv_command_blocking(duration));
            Ok(received.map(|inner| PyControlCommand { inner }))
        }

        /// Stop listening and join the background thread. Idempotent.
        fn close(&mut self) {
            if let Some(server) = self.inner.take() {
                server.shutdown();
            }
        }

        fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
        fn __exit__(
            &mut self,
            _exc_type: Option<Py<PyAny>>,
            _exc_value: Option<Py<PyAny>>,
            _traceback: Option<Py<PyAny>>,
        ) -> bool {
            self.close();
            false
        }

        fn __repr__(&self) -> String {
            match &self.inner {
                Some(server) => format!(
                    "Publisher(address='{}', clients={})",
                    self.address,
                    server.n_clients()
                ),
                None => format!("Publisher(address='{}', closed)", self.address),
            }
        }
    }

    impl PyPublisher {
        /// The live server, or a clear error once `close()` has run. Every
        /// method goes through here so a closed server never looks idle.
        fn server(&self) -> PyResult<&Publisher> {
            self.inner.as_ref().ok_or_else(|| {
                PyValueError::new_err("Publisher is closed; bind a new one to stream again")
            })
        }
    }
}

/// Register `molpy.stream`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyControlCommand>()?;
    #[cfg(not(target_arch = "wasm32"))]
    m.add_class::<PyPublisherConfig>()?;
    #[cfg(not(target_arch = "wasm32"))]
    m.add_class::<PyPublisher>()?;
    Ok(())
}
