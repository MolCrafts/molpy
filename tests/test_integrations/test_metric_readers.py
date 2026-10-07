"""The metric readers molpy publishes for any molcrafts viewer.

molrs parses these formats; molpy publishes the readers that turn them into
metric records, so this is where that mapping is proved. Nothing here imports
a host: a reader is matched structurally, and these tests check the
shape a host relies on — ``format`` / ``sniff`` / ``read`` plus the optional
``patterns`` and ``tailable`` hints — as well as the mapping itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy.integrations.metric_readers import (
    LammpsLogReader,
    MlpJsonlReader,
    MrecReader,
)

LAMMPS_LOG = """LAMMPS (2 Aug 2023)
Per MPI rank memory allocation (min/avg/max) = 3.5 | 3.5 | 3.5 Mbytes
   Step          Temp          Density
         0   300.00        1.05
      1000   310.00        1.04
      2000   320.00        1.03
      3000   330.00        1.02
Loop time of 4.21 on 4 procs for 3000 steps with 1000 atoms
Total wall time: 0:00:04
"""


class Request:
    """The sampling policy a viewer hands down. Matched by shape, not import."""

    def __init__(self, *, since=0, stride=1, limit=None, metric_type=None, keys=()):
        self.since = since
        self.stride = stride
        self.limit = limit
        self.metric_type = metric_type
        self.keys = keys


@pytest.fixture
def lammps_log(tmp_path: Path) -> Path:
    path = tmp_path / "log.lammps"
    path.write_text(LAMMPS_LOG, encoding="utf-8")
    return path


class TestTheContractShape:
    """A host looks these up by name; missing one silently disables a format."""

    @pytest.mark.parametrize(
        ("reader", "format_id", "tailable"),
        [
            (LammpsLogReader(), "lammps_log", False),
            (MrecReader(), "mrec", False),
            (MlpJsonlReader(), "mlp_jsonl", True),
        ],
    )
    def test_each_reader_declares_what_a_host_reads(self, reader, format_id, tailable):
        assert reader.format == format_id
        assert reader.tailable is tailable
        assert reader.patterns and all(isinstance(p, str) for p in reader.patterns)
        assert callable(reader.sniff)
        assert callable(reader.read)

    def test_no_reader_imports_a_host(self):
        """The arrow points one way: molpy publishes, a host consumes.

        Naming molexp in prose is fine — explaining who consumes this is the
        point. Importing it would invert the dependency.
        """
        import ast

        import molpy.integrations.metric_readers as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"molexp", "molplot", "molvis"}, imported


class TestLammpsLogReader:
    def test_claims_a_log_with_a_thermo_table(self, lammps_log: Path):
        assert LammpsLogReader().sniff(lammps_log)

    def test_rejects_a_log_that_never_reached_thermo(self, tmp_path: Path):
        path = tmp_path / "log.lammps"
        path.write_text("LAMMPS (2 Aug 2023)\nERROR: bad input\n", encoding="utf-8")
        assert not LammpsLogReader().sniff(path)

    def test_rejects_an_unrelated_log_despite_the_suffix(self, tmp_path: Path):
        path = tmp_path / "leap.log"
        path.write_text("Welcome to LEaP!\n", encoding="utf-8")
        assert not LammpsLogReader().sniff(path)

    def test_one_record_per_column_per_row_carrying_the_step(self, lammps_log: Path):
        records = list(LammpsLogReader().read(lammps_log, source="log.lammps"))
        assert {r["k"] for r in records} == {"lammps/Temp", "lammps/Density"}
        temps = [r for r in records if r["k"] == "lammps/Temp"]
        assert [r["s"] for r in temps] == [0.0, 1000.0, 2000.0, 3000.0]
        assert [r["v"] for r in temps] == [300.0, 310.0, 320.0, 330.0]

    def test_step_is_never_emitted_as_its_own_series(self, lammps_log: Path):
        records = list(LammpsLogReader().read(lammps_log, source="log.lammps"))
        assert "lammps/Step" not in {r["k"] for r in records}

    def test_wall_time_is_marked_as_ingest_not_measurement(self, lammps_log: Path):
        """LAMMPS records steps, not clock time; inventing timestamps would
        fabricate data."""
        records = list(LammpsLogReader().read(lammps_log, source="log.lammps"))
        assert all(r["tags"]["wall_time_source"] == "ingest" for r in records)
        assert all("w" not in r for r in records)

    def test_stride_is_honoured_and_keeps_every_series(self, lammps_log: Path):
        records = list(
            LammpsLogReader().read(
                lammps_log, source="log.lammps", request=Request(stride=2)
            )
        )
        assert {r["k"] for r in records} == {"lammps/Temp", "lammps/Density"}
        temps = [r for r in records if r["k"] == "lammps/Temp"]
        assert [r["s"] for r in temps] == [0.0, 2000.0]

    def test_limit_stops_the_read(self, lammps_log: Path):
        records = list(
            LammpsLogReader().read(
                lammps_log, source="log.lammps", request=Request(limit=3)
            )
        )
        assert len(records) == 3

    def test_key_filter_selects_one_series(self, lammps_log: Path):
        records = list(
            LammpsLogReader().read(
                lammps_log,
                source="log.lammps",
                request=Request(keys=("lammps/Density",)),
            )
        )
        assert {r["k"] for r in records} == {"lammps/Density"}

    def test_the_source_tag_travels_with_every_record(self, lammps_log: Path):
        records = list(LammpsLogReader().read(lammps_log, source="out/log.lammps"))
        assert all(r["tags"]["source"] == "out/log.lammps" for r in records)


class TestMlpJsonlReader:
    @pytest.fixture
    def wal(self, tmp_path: Path) -> Path:
        path = tmp_path / "metrics.mlp.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"t": "scalar", "k": key, "v": float(step)})
                for step in range(6)
                for key in ("energy", "temp")
            ),
            encoding="utf-8",
        )
        return path

    def test_claims_only_the_mlp_suffix(self, wal: Path, tmp_path: Path):
        assert MlpJsonlReader().sniff(wal)
        other = tmp_path / "notes.jsonl"
        other.write_text("{}", encoding="utf-8")
        assert not MlpJsonlReader().sniff(other)

    def test_records_pass_through_with_their_source_tagged(self, wal: Path):
        records = list(MlpJsonlReader().read(wal, source="out/metrics.mlp.jsonl"))
        assert len(records) == 12
        assert all(r["tags"]["source"] == "out/metrics.mlp.jsonl" for r in records)

    def test_stride_counts_per_series_not_per_line(self, wal: Path):
        """Series interleave line by line, so striding the file would keep one
        key and drop the other entirely."""
        records = list(
            MlpJsonlReader().read(wal, source="x", request=Request(stride=3))
        )
        assert {r["k"] for r in records} == {"energy", "temp"}
        assert len(records) == 4  # two samples of each of the two series

    def test_a_malformed_line_is_skipped_not_fatal(self, tmp_path: Path):
        path = tmp_path / "metrics.mlp.jsonl"
        path.write_text('{"t":"scalar","k":"a","v":1}\nnot json\n', encoding="utf-8")
        assert len(list(MlpJsonlReader().read(path, source="x"))) == 1


class TestMrecReader:
    def test_claims_the_mrec_suffix_only(self, tmp_path: Path):
        record = tmp_path / "growth.mrec"
        record.mkdir()
        assert MrecReader().sniff(record)
        assert not MrecReader().sniff(tmp_path / "growth.zarr")

    def test_emits_the_step_and_time_series(self, tmp_path: Path):
        frame = mp.Frame()
        frame["atoms"] = {"x": [0.0], "y": [0.0], "z": [0.0]}
        path = tmp_path / "run.mrec"
        mp.io.mrec.write_trajectory(
            path,
            mp.Trajectory(
                [frame, frame, frame],
                step=np.array([0, 10, 20]),
                time=np.array([0.0, 0.5, 1.0]),
            ),
        )
        records = list(MrecReader().read(path, source="run"))
        steps = [r["v"] for r in records if r["k"] == "mrec/step"]
        times = [r["v"] for r in records if r["k"] == "mrec/time"]
        assert steps == [0.0, 10.0, 20.0]
        assert times == [0.0, 0.5, 1.0]
        assert all(r["tags"] == {"source": "run"} for r in records)
