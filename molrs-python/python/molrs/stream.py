"""Live Frame streaming — ``molrs::stream``: :class:`Publisher`,
:class:`PublisherConfig` and :class:`ControlCommand`.

A producer binds a :class:`Publisher` (``Publisher.bind(address, config)``;
``Publisher.connect(url, config)`` dials a relay instead) and calls
``send(frame)`` once per simulation step. The call never blocks on the network: frames go through a
bounded buffer that drops the oldest payload when a viewer cannot keep up, so
a slow client slows nothing down. Viewers dial the socket and decode payloads
with :func:`molrs.io.read_wire_frame` (``encoding`` ``"msgpack"`` or
``"json"``, as the publisher's :class:`PublisherConfig` says); its inverse
:func:`molrs.io.write_wire_frame` is the encoding the publisher puts on the
wire. Like every reader and writer, those are :mod:`molrs.io`'s; this module
is the transport.

Traffic the other way is :class:`ControlCommand` — a viewer asking the producer
to pause, change rate, or restrict the streamed atom subset. Nothing here acts
on a command; the producer decides what it means.

:class:`Publisher` and :class:`PublisherConfig` open TCP sockets and are
therefore native-only, gated exactly as ``molrs::stream::publisher`` is. A
Pyodide build has :class:`ControlCommand` and no server — absent, rather than a stub that would
fail only once someone tried to connect.

Example
-------
Producer::

    import molrs

    config = molrs.stream.PublisherConfig(format="msgpack")
    with molrs.stream.Publisher.bind("127.0.0.1:8765", config) as server:
        for _ in range(n_steps):
            integrator.step()
            server.send(integrator.frame)
            cmd = server.recv_command()
            if cmd is not None and cmd.kind == "pause":
                ...
"""

from ._native import ControlCommand

__all__ = ["ControlCommand"]

try:  # native only — see the module docstring
    from ._native import (  # noqa: F401 — appended to __all__ below
        Publisher,
        PublisherConfig,
    )
except ImportError:  # pragma: no cover — Pyodide build
    pass
else:
    __all__ += ["Publisher", "PublisherConfig"]
