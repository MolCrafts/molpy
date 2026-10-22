//! Native WebSocket server that broadcasts serialized [`Frame`]s to clients.

use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use std::sync::mpsc::{self, SyncSender, TryRecvError};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, oneshot};
use tokio_tungstenite::tungstenite::Bytes;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;

use crate::core::Frame;
use crate::stream::{FrameEncoding, StreamError, encode_frame};

use super::control::ControlCommand;

/// Configuration for a [`Publisher`].
#[derive(Debug, Clone)]
pub struct PublisherConfig {
    /// Wire encoding for outbound frames (default: MessagePack).
    pub format: FrameEncoding,
    /// Capacity of the simulation→network frame buffer (default: 4).
    ///
    /// When full, [`Publisher::send`] drops the oldest buffered frame so the
    /// producer never blocks.
    pub buffer_size: usize,
    /// Reserved maximum stream rate in Hz. Not enforced (no-op).
    pub max_frame_rate: f64,
    /// Shared secret a client must present before it receives anything.
    ///
    /// `None` (the default) accepts every connection — right for a loopback
    /// bind you control, wrong for anything reachable by another user. A
    /// listening socket with no token lets whoever can reach the port read the
    /// whole trajectory *and* inject control commands, so set this whenever the
    /// bind address is not `127.0.0.1`.
    ///
    /// The handshake matches the one MolVis already speaks, so a client learns
    /// one shape rather than two:
    ///
    /// ```text
    /// client → server   {"type":"hello","token":"…"}
    /// server → client   {"type":"ready"}            (✓)
    ///                   close(1008, "auth")         (✗ missing / wrong token)
    /// ```
    /// WebSocket upgrades and token exchanges each time out after five seconds.
    pub token: Option<String>,
}

impl Default for PublisherConfig {
    fn default() -> Self {
        Self {
            format: FrameEncoding::MessagePack,
            buffer_size: 4,
            max_frame_rate: 0.0,
            token: None,
        }
    }
}

/// Error returned by [`Publisher::send`].
#[derive(Debug)]
pub enum SendError {
    /// Frame encoding failed.
    Encode(StreamError),
    /// The server has been shut down or the background thread exited.
    Closed,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::Encode(e) => write!(f, "frame encode error: {e}"),
            SendError::Closed => write!(f, "frame server closed"),
        }
    }
}

impl std::error::Error for SendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SendError::Encode(e) => Some(e),
            SendError::Closed => None,
        }
    }
}

impl From<StreamError> for SendError {
    fn from(e: StreamError) -> Self {
        SendError::Encode(e)
    }
}

/// What the publisher handle and its background thread both hold.
struct PublisherState {
    format: FrameEncoding,
    /// Simulation → clients (payload already encoded). A broadcast channel
    /// overwrites its oldest entry when full, which is the drop policy.
    frame_tx: Mutex<Option<broadcast::Sender<Bytes>>>,
    cmd_rx: Mutex<mpsc::Receiver<ControlCommand>>,
    n_clients: Arc<AtomicUsize>,
    local_addr: Option<SocketAddr>,
    shutting_down: AtomicBool,
    join: Mutex<Option<JoinHandle<()>>>,
    shutdown_tx: Mutex<Option<oneshot::Sender<()>>>,
}

/// Publishes over a WebSocket and collects control commands back.
///
/// The accept loop and per-client I/O run on a background `std::thread` that
/// owns a multi-thread tokio runtime. The simulation loop stays synchronous:
/// [`Publisher::send`] never blocks on network writes; when the bounded
/// broadcast buffer is full the oldest payload is dropped.
///
/// Named for neither its socket nor its payload, because it commits to
/// neither. It [`bind`](Self::bind)s and waits to be dialed, or
/// [`connect`](Self::connect)s out itself — that is the constructor's business,
/// not the type's. And everything here except [`send`](Self::send) is
/// payload-agnostic: the bounded channel, the drop-oldest policy, the
/// broadcast fan-out, the token handshake, the control channel. A `Frame` is
/// what it carries today, not what it is.
///
/// Clone freely: all clones share the same state. Call
/// [`Publisher::shutdown`] to stop the background thread, or drop the last
/// clone to join on exit.
#[derive(Clone)]
pub struct Publisher {
    state: Arc<PublisherState>,
}

impl Publisher {
    /// Bind a WebSocket server on `addr` and publish to every client that
    /// dials it.
    ///
    /// Use `"127.0.0.1:0"` to pick an ephemeral port, then read
    /// [`local_addr`](Self::local_addr). `PublisherConfig::default()` is the
    /// loopback-only setting: it demands no [`token`](PublisherConfig::token).
    pub fn bind(addr: impl Into<String>, config: PublisherConfig) -> io::Result<Self> {
        let addr = addr.into();
        let buffer_size = config.buffer_size.max(1);
        let format = config.format;
        let token = config.token;

        let (frame_tx, _) = broadcast::channel::<Bytes>(buffer_size);
        let bcast_tx = frame_tx.clone();
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<ControlCommand>(64);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<io::Result<SocketAddr>>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let n_clients = Arc::new(AtomicUsize::new(0));
        let client_count_thread = Arc::clone(&n_clients);

        let join = std::thread::Builder::new()
            .name("molrs-frame-publisher".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .worker_threads(2)
                    .thread_name("molrs-net")
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx.send(Err(io::Error::other(e)));
                        return;
                    }
                };

                rt.block_on(async move {
                    let listener = match TcpListener::bind(&addr).await {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    let local = match listener.local_addr() {
                        Ok(a) => a,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    let _ = ready_tx.send(Ok(local));

                    run_bound(
                        listener,
                        bcast_tx,
                        cmd_tx,
                        client_count_thread,
                        format,
                        token,
                        shutdown_rx,
                    )
                    .await;
                });
            })?;

        let local_addr = ready_rx
            .recv()
            .map_err(|_| io::Error::other("frame server thread exited before bind"))??;

        Ok(Publisher {
            state: Arc::new(PublisherState {
                format,
                frame_tx: Mutex::new(Some(frame_tx)),
                cmd_rx: Mutex::new(cmd_rx),
                n_clients,
                local_addr: Some(local_addr),
                shutting_down: AtomicBool::new(false),
                join: Mutex::new(Some(join)),
                shutdown_tx: Mutex::new(Some(shutdown_tx)),
            }),
        })
    }

    /// Dial `url` and publish to whatever is listening there.
    ///
    /// The mirror of [`bind`](Self::bind). Same protocol, same wire format,
    /// same control channel — the only difference is which end opens the TCP
    /// connection, which has never had anything to do with which end sends
    /// frames.
    ///
    /// Reach for this when the producer cannot be dialed: a compute node behind
    /// a firewall can open an outbound connection to a collector even when
    /// nothing can open one to it. Prefer [`bind`](Self::bind) otherwise — a
    /// bound producer outlives its viewers, so a viewer can reattach after a
    /// reload and several can watch at once.
    ///
    /// Note that a browser can only dial, so the listener at `url` has to be a
    /// native host, not a page.
    ///
    /// [`PublisherConfig::token`] is *presented* here rather than demanded: this
    /// end is the client of the handshake when it dials. [`local_addr`] returns
    /// `None` for a dialed publisher — there is no address for anyone to
    /// connect to.
    ///
    /// [`local_addr`]: Self::local_addr
    pub fn connect(url: impl Into<String>, config: PublisherConfig) -> io::Result<Self> {
        let url = url.into();
        let buffer_size = config.buffer_size.max(1);
        let format = config.format;
        let token = config.token;

        let (frame_tx, _) = broadcast::channel::<Bytes>(buffer_size);
        let bcast_tx = frame_tx.clone();
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<ControlCommand>(64);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let n_clients = Arc::new(AtomicUsize::new(0));
        let client_count_thread = Arc::clone(&n_clients);

        let join = std::thread::Builder::new()
            .name("molrs-frame-publisher".into())
            .spawn(move || {
                let Ok(rt) = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .worker_threads(2)
                    .thread_name("molrs-stream")
                    .build()
                else {
                    return;
                };
                rt.block_on(run_dialed(
                    url,
                    bcast_tx,
                    cmd_tx,
                    client_count_thread,
                    format,
                    token,
                    shutdown_rx,
                ));
            })?;

        Ok(Publisher {
            state: Arc::new(PublisherState {
                format,
                frame_tx: Mutex::new(Some(frame_tx)),
                cmd_rx: Mutex::new(cmd_rx),
                n_clients,
                local_addr: None,
                shutting_down: AtomicBool::new(false),
                join: Mutex::new(Some(join)),
                shutdown_tx: Mutex::new(Some(shutdown_tx)),
            }),
        })
    }

    /// Local socket address the server is listening on.
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.state.local_addr
    }

    /// Number of currently connected WebSocket clients.
    pub fn n_clients(&self) -> usize {
        self.state.n_clients.load(Ordering::Relaxed)
    }

    /// Encode `frame` and enqueue it for broadcast.
    ///
    /// Never blocks on network I/O. If the internal buffer is full, the oldest
    /// pending frame is dropped so this call returns promptly.
    pub fn send(&self, frame: &Frame) -> Result<(), SendError> {
        let bytes = encode_frame(frame, self.state.format)?;
        self.send_bytes(Bytes::from(bytes))
    }

    /// Wait for the next control command from any client.
    ///
    /// Returns `None` when the server is shutting down or the command channel
    /// is closed. Polls with a short sleep so it is safe to call from any tokio
    /// runtime (not only the server's background runtime).
    pub async fn recv_command(&self) -> Option<ControlCommand> {
        loop {
            let polled = match self.state.cmd_rx.lock() {
                Ok(rx) => rx.try_recv(),
                Err(_) => return None,
            };
            match polled {
                Ok(cmd) => return Some(cmd),
                Err(TryRecvError::Disconnected) => return None,
                Err(TryRecvError::Empty) => {
                    if self.state.shutting_down.load(Ordering::Acquire) {
                        return None;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        }
    }

    /// Like [`recv_command`](Self::recv_command) but returns `None` if `timeout` elapses first.
    pub async fn recv_command_timeout(&self, timeout: Duration) -> Option<ControlCommand> {
        tokio::time::timeout(timeout, self.recv_command())
            .await
            .unwrap_or_default()
    }

    /// Blocking counterpart to [`recv_command_timeout`](Self::recv_command_timeout).
    ///
    /// The producer side of this server is synchronous by design — [`send`]
    /// never blocks and never needs a runtime — so polling for a viewer's
    /// command must not force the caller to own one either. A zero `timeout`
    /// polls and returns immediately.
    ///
    /// Returns `None` on timeout and on a closed command channel, matching
    /// [`recv_command`](Self::recv_command).
    ///
    /// [`send`]: Self::send
    pub fn recv_command_blocking(&self, timeout: Duration) -> Option<ControlCommand> {
        let rx = self.state.cmd_rx.lock().ok()?;
        if timeout.is_zero() {
            return rx.try_recv().ok();
        }
        rx.recv_timeout(timeout).ok()
    }

    /// Signal the accept loop to stop and join the background thread.
    pub fn shutdown(self) {
        self.request_shutdown();
        if let Ok(mut guard) = self.state.join.lock()
            && let Some(handle) = guard.take()
        {
            let _ = handle.join();
        }
    }

    fn request_shutdown(&self) {
        self.state.shutting_down.store(true, Ordering::Release);
        if let Ok(mut guard) = self.state.frame_tx.lock() {
            *guard = None;
        }
        if let Ok(mut guard) = self.state.shutdown_tx.lock()
            && let Some(tx) = guard.take()
        {
            let _ = tx.send(());
        }
    }

    fn send_bytes(&self, bytes: Bytes) -> Result<(), SendError> {
        let tx_guard = self.state.frame_tx.lock().map_err(|_| SendError::Closed)?;
        let tx = tx_guard.as_ref().ok_or(SendError::Closed)?;
        // A full broadcast buffer overwrites its oldest entry; a send with no
        // subscriber is simply a frame nobody was attached to receive.
        let _ = tx.send(bytes);
        Ok(())
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        // Only the last Arc clone should join; earlier clones leave the server running.
        if Arc::strong_count(&self.state) > 1 {
            return;
        }
        self.request_shutdown();
        if let Ok(mut guard) = self.state.join.lock()
            && let Some(handle) = guard.take()
        {
            let _ = handle.join();
        }
    }
}

// --- Background runtime -------------------------------------------------------

/// Bound both the WebSocket upgrade and the application token handshake.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

async fn run_bound(
    listener: TcpListener,
    bcast_tx: broadcast::Sender<Bytes>,
    cmd_tx: SyncSender<ControlCommand>,
    n_clients: Arc<AtomicUsize>,
    format: FrameEncoding,
    token: Option<String>,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    loop {
        tokio::select! {
            _ = &mut shutdown_rx => break,
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else { continue };
                let bcast_rx = bcast_tx.subscribe();
                let cmd_tx = cmd_tx.clone();
                let n_clients = Arc::clone(&n_clients);
                let token = token.clone();
                tokio::spawn(async move {
                    let Ok(Ok(ws)) = tokio::time::timeout(
                        HANDSHAKE_TIMEOUT, tokio_tungstenite::accept_async(stream),
                    ).await else {
                        return;
                    };
                    n_clients.fetch_add(1, Ordering::Relaxed);
                    handle_client(ws, bcast_rx, cmd_tx, format, token).await;
                    n_clients.fetch_sub(1, Ordering::Relaxed);
                });
            }
        }
    }
}

use tokio_tungstenite::WebSocketStream;

type WsStream = WebSocketStream<tokio::net::TcpStream>;

/// A handshake message: `hello` carries a token, `ready` acknowledges it.
#[derive(serde::Deserialize)]
struct Hello {
    #[serde(rename = "type")]
    kind: String,
    token: Option<String>,
}

/// Gate a freshly accepted socket on the shared secret.
///
/// Returns `true` once the client is cleared to receive frames. A server with
/// no token configured clears everyone without exchanging anything, so an
/// unauthenticated client of an unauthenticated server needs no handshake code
/// at all.
///
/// The comparison is not constant-time. The secret is a per-run bearer token on
/// a socket the operator chose to expose, not a stored credential, and the
/// dominant risk here is an unauthenticated bind — not timing recovery of a
/// token an attacker must already be able to reach.
async fn authenticate(
    write: &mut futures_util::stream::SplitSink<WsStream, Message>,
    read: &mut futures_util::stream::SplitStream<WsStream>,
    expected: Option<&str>,
) -> bool {
    let Some(expected) = expected else {
        return true;
    };

    let offered = match read.next().await {
        Some(Ok(Message::Text(text))) => serde_json::from_str::<Hello>(text.as_ref())
            .ok()
            .filter(|h| h.kind == "hello")
            .and_then(|h| h.token),
        _ => None,
    };

    if offered.as_deref() != Some(expected) {
        let _ = write
            .send(Message::Close(Some(CloseFrame {
                code: CloseCode::Policy,
                reason: "auth".into(),
            })))
            .await;
        return false;
    }

    write
        .send(Message::Text("{\"type\":\"ready\"}".into()))
        .await
        .is_ok()
}

/// Dial `url`, publish until the socket dies, then dial again.
///
/// Redialling is not optional politeness: the collector is a separate process
/// and the producer is the long-running one, so a collector restart must not
/// end the run. `send` keeps dropping the oldest frame while nothing is
/// attached, exactly as it does for a bound publisher with no viewers.
async fn run_dialed(
    url: String,
    bcast_tx: broadcast::Sender<Bytes>,
    cmd_tx: SyncSender<ControlCommand>,
    n_clients: Arc<AtomicUsize>,
    format: FrameEncoding,
    token: Option<String>,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    loop {
        let dialed = tokio::select! {
            _ = &mut shutdown_rx => break,
            result = tokio::time::timeout(HANDSHAKE_TIMEOUT, tokio_tungstenite::connect_async(&url)) => result,
        };

        // A failed dial is the expected case while the collector is down; the
        // backoff below is the whole response to it.
        if let Ok(Ok((ws, _))) = dialed {
            {
                let (mut write, mut read) = ws.split();
                let authenticated = tokio::select! {
                    _ = &mut shutdown_rx => break,
                    result = tokio::time::timeout(HANDSHAKE_TIMEOUT, present_token(&mut write, &mut read, token.as_deref())) => result.unwrap_or(false),
                };
                if authenticated {
                    n_clients.fetch_add(1, Ordering::Relaxed);
                    let bcast_rx = bcast_tx.subscribe();
                    tokio::select! {
                        _ = &mut shutdown_rx => {
                            n_clients.fetch_sub(1, Ordering::Relaxed);
                            break;
                        }
                        _ = pump(write, read, bcast_rx, cmd_tx.clone(), format) => {}
                    }
                    n_clients.fetch_sub(1, Ordering::Relaxed);
                }
            }
        }

        // Back off so a collector that is down does not become a busy loop.
        tokio::select! {
            _ = &mut shutdown_rx => break,
            _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => {}
        }
    }

    drop(bcast_tx);
}

/// Present the shared secret, as the dialing end.
///
/// The mirror of [`authenticate`]. When this publisher dials out, it is the
/// client of the WebSocket handshake, so it offers the token instead of
/// demanding one. The message shape is identical in both directions, which is
/// the point of having one handshake rather than two.
async fn present_token<S>(
    write: &mut futures_util::stream::SplitSink<WebSocketStream<S>, Message>,
    read: &mut futures_util::stream::SplitStream<WebSocketStream<S>>,
    token: Option<&str>,
) -> bool
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let Some(token) = token else {
        return true;
    };
    let hello = serde_json::json!({ "type": "hello", "token": token }).to_string();
    if write.send(Message::Text(hello.into())).await.is_err() {
        return false;
    }
    matches!(read.next().await, Some(Ok(Message::Text(t))) if
        serde_json::from_str::<Hello>(&t).is_ok_and(|message| message.kind == "ready"))
}

async fn handle_client(
    ws: WsStream,
    bcast_rx: broadcast::Receiver<Bytes>,
    cmd_tx: SyncSender<ControlCommand>,
    format: FrameEncoding,
    token: Option<String>,
) {
    let (mut write, mut read) = ws.split();

    // Nothing goes out before the client is cleared — not one frame.
    if !tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        authenticate(&mut write, &mut read, token.as_deref()),
    )
    .await
    .unwrap_or(false)
    {
        return;
    }
    pump(write, read, bcast_rx, cmd_tx, format).await;
}

/// Frames out, control commands in, until either side goes away.
///
/// Identical whichever end dialed: once the socket is open a WebSocket is
/// symmetric, and connection direction has never had anything to do with data
/// direction. This is why `bind` and `connect` are one protocol and not two.
async fn pump<S>(
    mut write: futures_util::stream::SplitSink<WebSocketStream<S>, Message>,
    mut read: futures_util::stream::SplitStream<WebSocketStream<S>>,
    mut bcast_rx: broadcast::Receiver<Bytes>,
    cmd_tx: SyncSender<ControlCommand>,
    format: FrameEncoding,
) where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    loop {
        tokio::select! {
            frame = bcast_rx.recv() => {
                match frame {
                    Ok(payload) => {
                        let msg = match format {
                            FrameEncoding::MessagePack => Message::Binary(payload),
                            FrameEncoding::Json => {
                                match String::from_utf8(payload.to_vec()) {
                                    Ok(s) => Message::Text(s.into()),
                                    Err(_) => continue,
                                }
                            }
                        };
                        if write.send(msg).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            incoming = read.next() => {
                match incoming {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(cmd) = serde_json::from_str::<ControlCommand>(text.as_ref()) {
                            let _ = cmd_tx.try_send(cmd);
                        }
                    }
                    Some(Ok(Message::Binary(bin))) => {
                        if let Ok(cmd) = rmp_serde::from_slice::<ControlCommand>(bin.as_ref()) {
                            let _ = cmd_tx.try_send(cmd);
                        }
                    }
                    Some(Ok(Message::Ping(p))) => {
                        let _ = write.send(Message::Pong(p)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Block;
    use crate::op::{F, I};
    use crate::stream::decode_msgpack_frame;
    use futures_util::{SinkExt, StreamExt};
    use ndarray::Array1;
    use tokio_tungstenite::connect_async;

    fn sample_frame(tag: i32) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F, tag as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.0 as F, 0.0 as F, 1.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "serial",
                Array1::from_vec(vec![1 as I, 2 as I, 3 as I]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.meta.insert("tag", tag);
        frame
    }

    async fn wait_clients(server: &Publisher, n: usize) {
        for _ in 0..100 {
            if server.n_clients() == n {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "timed out waiting for {n} clients (have {})",
            server.n_clients()
        );
    }

    #[tokio::test]
    async fn bind_and_client_connects() {
        let server = Publisher::bind("127.0.0.1:0", PublisherConfig::default()).expect("bind");
        let url = format!("ws://{}", server.local_addr().expect("bound"));

        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;
        assert_eq!(server.n_clients(), 1);

        ws.close(None).await.ok();
        server.shutdown();
    }

    #[tokio::test]
    async fn an_incomplete_handshake_does_not_block_other_clients_or_shutdown() {
        let server = Publisher::bind("127.0.0.1:0", PublisherConfig::default()).unwrap();
        let addr = server.local_addr().unwrap();
        let stalled = tokio::net::TcpStream::connect(addr).await.unwrap();
        let connected = tokio::time::timeout(
            Duration::from_secs(1),
            connect_async(format!("ws://{addr}")),
        )
        .await;
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let shutdown = std::thread::spawn(move || {
            server.shutdown();
            done_tx.send(()).unwrap();
        });
        let stopped = done_rx.recv_timeout(Duration::from_secs(1));
        // Release the peer even on failure, so a regression cannot hang the test process.
        drop(stalled);
        shutdown.join().unwrap();
        assert!(
            connected.is_ok(),
            "another client was blocked by the stalled handshake"
        );
        assert!(stopped.is_ok(), "shutdown waited for the stalled handshake");
    }

    #[tokio::test]
    async fn shutdown_cancels_a_dialed_token_handshake() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let publisher = Publisher::connect(
            format!("ws://{}", listener.local_addr().unwrap()),
            PublisherConfig {
                token: Some("secret".into()),
                ..PublisherConfig::default()
            },
        )
        .unwrap();
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        let hello = ws.next().await.unwrap().unwrap();
        assert!(matches!(hello, Message::Text(_)));
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let shutdown = std::thread::spawn(move || {
            publisher.shutdown();
            done_tx.send(()).unwrap();
        });
        let stopped = done_rx.recv_timeout(Duration::from_secs(1));
        drop(ws);
        shutdown.join().unwrap();
        assert!(
            stopped.is_ok(),
            "shutdown waited for a token acknowledgement"
        );
    }

    #[tokio::test]
    async fn a_token_acknowledgement_must_be_a_ready_message() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = async {
            let (ws, _) = connect_async(format!("ws://{addr}")).await.unwrap();
            let (mut write, mut read) = ws.split();
            present_token(&mut write, &mut read, Some("secret")).await
        };
        let collector = async {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            ws.next().await.unwrap().unwrap();
            ws.send(Message::Text(r#"{"type":"not_ready"}"#.into()))
                .await
                .unwrap();
        };
        let (accepted, ()) = tokio::join!(client, collector);
        assert!(!accepted);
    }

    #[tokio::test]
    async fn send_frame_received_by_client() {
        let server = Publisher::bind("127.0.0.1:0", PublisherConfig::default()).expect("bind");
        let url = format!("ws://{}", server.local_addr().expect("bound"));
        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;

        let frame = sample_frame(7);
        server.send(&frame).expect("send");

        let msg = tokio::time::timeout(Duration::from_secs(2), ws.next())
            .await
            .expect("timeout")
            .expect("stream closed")
            .expect("ws error");

        let bytes = match msg {
            Message::Binary(b) => b.to_vec(),
            Message::Text(t) => t.as_bytes().to_vec(),
            other => panic!("unexpected message: {other:?}"),
        };
        let decoded = decode_msgpack_frame(&bytes).expect("decode");
        assert!(decoded.contains_key("atoms"));
        let x = decoded["atoms"]
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((x[2] - 7.0).abs() < 1e-12);

        ws.close(None).await.ok();
        server.shutdown();
    }

    #[tokio::test]
    async fn client_pause_command_received() {
        let server = Publisher::bind("127.0.0.1:0", PublisherConfig::default()).expect("bind");
        let url = format!("ws://{}", server.local_addr().expect("bound"));
        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;

        let payload = serde_json::to_string(&ControlCommand::Pause).unwrap();
        ws.send(Message::Text(payload.into()))
            .await
            .expect("send cmd");

        let cmd = server
            .recv_command_timeout(Duration::from_secs(2))
            .await
            .expect("expected Pause");
        assert_eq!(cmd, ControlCommand::Pause);

        ws.close(None).await.ok();
        server.shutdown();
    }

    #[tokio::test]
    async fn buffer_full_send_does_not_block() {
        let server = Publisher::bind(
            "127.0.0.1:0",
            PublisherConfig {
                format: FrameEncoding::MessagePack,
                buffer_size: 1,
                max_frame_rate: 0.0,
                token: None,
            },
        )
        .expect("bind");

        let start = std::time::Instant::now();
        for i in 0..3 {
            server.send(&sample_frame(i)).expect("send");
        }
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "send blocked for {:?}",
            start.elapsed()
        );
        server.shutdown();
    }

    #[tokio::test]
    async fn buffer_full_client_eventually_gets_latest() {
        let server = Publisher::bind(
            "127.0.0.1:0",
            PublisherConfig {
                format: FrameEncoding::MessagePack,
                buffer_size: 1,
                max_frame_rate: 0.0,
                token: None,
            },
        )
        .expect("bind");
        let url = format!("ws://{}", server.local_addr().expect("bound"));
        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;

        for i in 1..=3 {
            server.send(&sample_frame(i)).expect("send");
        }

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let mut saw_latest = false;
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(200), ws.next()).await {
                Ok(Some(Ok(Message::Binary(b)))) => {
                    if let Ok(decoded) = decode_msgpack_frame(b.as_ref())
                        && let Some(x) = decoded
                            .get("atoms")
                            .and_then(|a| a.get("x").and_then(|c| c.get::<f64>()))
                        && (x[2] - 3.0).abs() < 1e-12
                    {
                        saw_latest = true;
                        break;
                    }
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(_))) | Ok(None) => break,
                Err(_) => {}
            }
        }
        assert!(saw_latest, "client never received the latest frame (tag 3)");

        ws.close(None).await.ok();
        server.shutdown();
    }

    fn token_server(token: &str) -> Publisher {
        Publisher::bind(
            "127.0.0.1:0",
            PublisherConfig {
                token: Some(token.to_string()),
                ..PublisherConfig::default()
            },
        )
        .expect("bind")
    }

    #[tokio::test]
    async fn a_correct_token_is_cleared_and_then_receives_frames() {
        let server = token_server("s3cret");
        let url = format!("ws://{}", server.local_addr().expect("bound"));
        let (mut ws, _) = connect_async(&url).await.expect("connect");

        ws.send(Message::Text(r#"{"type":"hello","token":"s3cret"}"#.into()))
            .await
            .expect("hello");
        let ready = ws.next().await.expect("ready").expect("ready ok");
        assert_eq!(ready.into_text().unwrap().as_str(), r#"{"type":"ready"}"#);

        wait_clients(&server, 1).await;
        server.send(&sample_frame(7)).expect("send");

        let msg = ws.next().await.expect("frame").expect("frame ok");
        let frame = decode_msgpack_frame(&msg.into_data()).expect("decode");
        let x = frame["atoms"]
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((x[2] - 7.0).abs() < 1e-12);
    }

    /// The gate: a client that never proves the token must not receive a single
    /// frame, even one published after it connected.
    #[tokio::test]
    async fn a_wrong_token_gets_no_frames_at_all() {
        let server = token_server("s3cret");
        let url = format!("ws://{}", server.local_addr().expect("bound"));
        let (mut ws, _) = connect_async(&url).await.expect("connect");

        ws.send(Message::Text(r#"{"type":"hello","token":"guess"}"#.into()))
            .await
            .expect("hello");
        server.send(&sample_frame(7)).expect("send");

        // Whatever arrives, none of it may be a frame.
        while let Some(Ok(msg)) = ws.next().await {
            match msg {
                Message::Close(_) => return,
                Message::Binary(_) | Message::Text(_) => {
                    panic!("an unauthenticated client received payload: {msg:?}")
                }
                _ => continue,
            }
        }
    }

    /// Skipping the handshake entirely is the same failure as getting it wrong —
    /// silence must not be read as consent.
    #[tokio::test]
    async fn saying_nothing_is_not_authentication() {
        let server = token_server("s3cret");
        let url = format!("ws://{}", server.local_addr().expect("bound"));
        let (mut ws, _) = connect_async(&url).await.expect("connect");

        drop(ws.send(Message::Close(None)).await);
        server.send(&sample_frame(7)).expect("send");

        while let Some(Ok(msg)) = ws.next().await {
            if matches!(msg, Message::Binary(_) | Message::Text(_)) {
                panic!("a silent client received payload: {msg:?}");
            }
        }
    }

    /// A tokenless server exchanges nothing — an existing client keeps working
    /// with no handshake code.
    #[tokio::test]
    async fn no_token_configured_means_no_handshake() {
        let server = Publisher::bind("127.0.0.1:0", PublisherConfig::default()).expect("bind");
        let url = format!("ws://{}", server.local_addr().expect("bound"));
        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;

        server.send(&sample_frame(3)).expect("send");
        let msg = ws.next().await.expect("frame").expect("frame ok");
        let frame = decode_msgpack_frame(&msg.into_data()).expect("decode");
        let x = frame["atoms"]
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    /// The point of `connect`: one protocol, two connection directions.
    /// A publisher that dialled out delivers frames exactly like a bound one.
    #[tokio::test]
    async fn a_dialed_publisher_delivers_frames() {
        // Stand in for the collector: a plain WebSocket listener.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listen");
        let addr = listener.local_addr().expect("addr");
        let accepted = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            tokio_tungstenite::accept_async(stream).await.expect("ws")
        });

        let publisher =
            Publisher::connect(format!("ws://{addr}"), PublisherConfig::default()).expect("dial");
        let mut ws = accepted.await.expect("join");

        // No address to hand out — nothing can dial a publisher that dialled.
        assert!(publisher.local_addr().is_none());

        for _ in 0..100 {
            if publisher.n_clients() == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        publisher.send(&sample_frame(11)).expect("send");

        let msg = ws.next().await.expect("frame").expect("frame ok");
        let frame = decode_msgpack_frame(&msg.into_data()).expect("decode");
        let x = frame["atoms"]
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((x[2] - 11.0).abs() < 1e-12);

        publisher.shutdown();
    }

    /// A collector that is not up *yet* must not end the run — the producer is
    /// the long-lived end, so it keeps dialing until something answers.
    ///
    /// The assertion has to be that it eventually connects. Checking only that
    /// `send` succeeds and `n_clients` is 0 while nothing listens proves
    /// nothing: that holds with the dial loop deleted entirely.
    #[tokio::test]
    async fn a_collector_that_starts_late_still_gets_the_stream() {
        // Reserve the port, then drop the listener so the first dials fail.
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe");
        let addr = probe.local_addr().expect("addr");
        drop(probe);

        let publisher =
            Publisher::connect(format!("ws://{addr}"), PublisherConfig::default()).expect("dial");

        // Be genuinely late. Without this the collector can bind before the
        // publisher's thread has built its runtime and dialed even once, so
        // the first attempt succeeds and the retry path is never exercised --
        // the test would then pass with the retry loop deleted.
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        assert_eq!(
            publisher.n_clients(),
            0,
            "nothing is listening yet, so no dial should have succeeded"
        );

        // The collector arrives late. Binding the same port can race with the
        // OS releasing it, so give it a few attempts.
        let mut listener = None;
        for _ in 0..50 {
            if let Ok(l) = tokio::net::TcpListener::bind(addr).await {
                listener = Some(l);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let listener = listener.expect("collector could not bind");

        let (stream, _) =
            tokio::time::timeout(std::time::Duration::from_secs(10), listener.accept())
                .await
                .expect("publisher never redialed")
                .expect("accept");
        let mut ws = tokio_tungstenite::accept_async(stream).await.expect("ws");

        for _ in 0..200 {
            if publisher.n_clients() == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        publisher.send(&sample_frame(5)).expect("send");

        let msg = tokio::time::timeout(std::time::Duration::from_secs(10), ws.next())
            .await
            .expect("no frame after reconnect")
            .expect("frame")
            .expect("frame ok");
        let frame = decode_msgpack_frame(&msg.into_data()).expect("decode");
        let x = frame["atoms"]
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((x[2] - 5.0).abs() < 1e-12);

        publisher.shutdown();
    }
}
