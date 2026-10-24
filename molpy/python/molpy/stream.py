"""Live Frame streaming — ``molrs::stream``: :class:`Publisher`,
:class:`PublisherConfig` and :class:`ControlCommand`.

A producer binds a :class:`Publisher` (``Publisher.bind(address, config)``;
``Publisher.connect(url, config)`` dials a relay instead) and calls
``send(frame)`` once per simulation step. The call never blocks on the network: frames go through a
bounded buffer that drops the oldest payload when a viewer cannot keep up, so
a slow client slows nothing down. Viewers dial the socket and decode payloads
with :func:`molpy.io.read_wire_frame` (``encoding`` ``"msgpack"`` or
``"json"``, as the publisher's :class:`PublisherConfig` says); its inverse
:func:`molpy.io.write_wire_frame` is the encoding the publisher puts on the
wire. Like every reader and writer, those are :mod:`molpy.io`'s; this module
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

    import molpy

    config = molpy.stream.PublisherConfig(format="msgpack")
    with molpy.stream.Publisher.bind("127.0.0.1:8765", config) as server:
        for _ in range(n_steps):
            integrator.step()
            server.send(integrator.frame)
            cmd = server.recv_command()
            if cmd is not None and cmd.kind == "pause":
                ...
"""

import sys

from molpy._core import ControlCommand

__all__ = [
    "ControlCommand",
]

# The Pyodide (emscripten) build has no sockets, so no publisher.
if sys.platform != "emscripten":
    from molpy._core import Publisher, PublisherConfig

    __all__ += ["Publisher", "PublisherConfig"]
