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
            (mp.io.lammps.LammpsLogMetricReader(), "lammps_log", False),
            (mp.io.mrec.MrecMetricReader(), "mrec", False),
            (mp.io.mlp_jsonl.MlpJsonlMetricReader(), "mlp_jsonl", True),
        ],
    )
    def test_each_reader_declares_what_a_host_reads(self, reader, format_id, tailable):
        assert reader.format == format_id
        assert reader.tailable is tailable
        assert reader.patterns and all(isinstance(p, str) for p in reader.patterns)
        assert callable(reader.sniff)
        assert callable(reader.read)

    def test_each_entry_point_names_a_reader(self):
        """molab finds a reader only through ``molcrafts.metric_readers``."""
        import importlib
        import tomllib

        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        group = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "entry-points"
        ]["molcrafts.metric_readers"]
        assert group == {
            "lammps_log": "molpy.io.lammps:LammpsLogMetricReader",
            "mrec": "molpy.io.mrec:MrecMetricReader",
            "mlp_jsonl": "molpy.io.mlp_jsonl:MlpJsonlMetricReader",
        }
        for format_id, target in group.items():
            module, name = target.split(":")
            reader = getattr(importlib.import_module(module), name)()
            assert reader.format == format_id


class TestLammpsLogReader:
    def test_claims_a_log_with_a_thermo_table(self, lammps_log: Path):
        assert mp.io.lammps.LammpsLogMetricReader().sniff(lammps_log)

    def test_claims_a_log_whose_run_starts_past_the_probe_window(self, tmp_path: Path):
        banner, _, rest = LAMMPS_LOG.partition("\n")
        assert banner.startswith("LAMMPS (")
        setup = "".join(f"# input line {i}\n" for i in range(8000))  # > 64 KiB
        path = tmp_path / "log.lammps"
        path.write_text(f"{banner}\n{setup}{rest}", encoding="utf-8")
        assert mp.io.lammps.LammpsLogMetricReader().sniff(path)

    def test_rejects_a_log_that_never_reached_thermo(self, tmp_path: Path):
        path = tmp_path / "log.lammps"
        path.write_text("LAMMPS (2 Aug 2023)\nERROR: bad input\n", encoding="utf-8")
        assert not mp.io.lammps.LammpsLogMetricReader().sniff(path)

    def test_rejects_an_unrelated_log_despite_the_suffix(self, tmp_path: Path):
        path = tmp_path / "leap.log"
        path.write_text("Welcome to LEaP!\n", encoding="utf-8")
        assert not mp.io.lammps.LammpsLogMetricReader().sniff(path)

    def test_one_record_per_column_per_row_carrying_the_step(self, lammps_log: Path):
        records = list(
            mp.io.lammps.LammpsLogMetricReader().read(lammps_log, source="log.lammps")
        )
        assert {r["k"] for r in records} == {"lammps/Temp", "lammps/Density"}
        temps = [r for r in records if r["k"] == "lammps/Temp"]
        assert [r["s"] for r in temps] == [0.0, 1000.0, 2000.0, 3000.0]
        assert [r["v"] for r in temps] == [300.0, 310.0, 320.0, 330.0]

    def test_step_is_never_emitted_as_its_own_series(self, lammps_log: Path):
        records = list(
            mp.io.lammps.LammpsLogMetricReader().read(lammps_log, source="log.lammps")
        )
        assert "lammps/Step" not in {r["k"] for r in records}

    def test_wall_time_is_marked_as_ingest_not_measurement(self, lammps_log: Path):
        """LAMMPS records steps, not clock time; inventing timestamps would
        fabricate data."""
        records = list(
            mp.io.lammps.LammpsLogMetricReader().read(lammps_log, source="log.lammps")
        )
        assert all(r["tags"]["wall_time_source"] == "ingest" for r in records)
        assert all("w" not in r for r in records)

    def test_stride_is_honoured_and_keeps_every_series(self, lammps_log: Path):
        records = list(
            mp.io.lammps.LammpsLogMetricReader().read(
                lammps_log, source="log.lammps", request=Request(stride=2)
            )
        )
        assert {r["k"] for r in records} == {"lammps/Temp", "lammps/Density"}
        temps = [r for r in records if r["k"] == "lammps/Temp"]
        assert [r["s"] for r in temps] == [0.0, 2000.0]

    def test_limit_stops_the_read(self, lammps_log: Path):
        records = list(
            mp.io.lammps.LammpsLogMetricReader().read(
                lammps_log, source="log.lammps", request=Request(limit=3)
            )
        )
        assert len(records) == 3

    def test_key_filter_selects_one_series(self, lammps_log: Path):
        records = list(
            mp.io.lammps.LammpsLogMetricReader().read(
                lammps_log,
                source="log.lammps",
                request=Request(keys=("lammps/Density",)),
            )
        )
        assert {r["k"] for r in records} == {"lammps/Density"}

    def test_the_source_tag_travels_with_every_record(self, lammps_log: Path):
        records = list(
            mp.io.lammps.LammpsLogMetricReader().read(
                lammps_log, source="out/log.lammps"
            )
        )
        assert all(r["tags"]["source"] == "out/log.lammps" for r in records)

    def test_without_a_step_column_s_is_the_row_ordinal_across_runs(
        self, tmp_path: Path
    ):
        table = (
            "Per MPI rank memory allocation (min/avg/max) = 3.5 | 3.5 | 3.5 Mbytes\n"
            "   Temp\n   300.0\n   310.0\n"
            "Loop time of 1.0 on 1 procs for 100 steps with 10 atoms\n"
        )
        path = tmp_path / "log.lammps"
        path.write_text(f"LAMMPS (2 Aug 2023)\n{table}{table}", encoding="utf-8")
        records = list(mp.io.lammps.LammpsLogMetricReader().read(path, source="x"))
        assert [r["s"] for r in records] == [0.0, 1.0, 2.0, 3.0]


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
        assert mp.io.mlp_jsonl.MlpJsonlMetricReader().sniff(wal)
        other = tmp_path / "notes.jsonl"
        other.write_text("{}", encoding="utf-8")
        assert not mp.io.mlp_jsonl.MlpJsonlMetricReader().sniff(other)

    def test_records_pass_through_with_their_source_tagged(self, wal: Path):
        records = list(
            mp.io.mlp_jsonl.MlpJsonlMetricReader().read(
                wal, source="out/metrics.mlp.jsonl"
            )
        )
        assert len(records) == 12
        assert all(r["tags"]["source"] == "out/metrics.mlp.jsonl" for r in records)

    def test_stride_counts_per_series_not_per_line(self, wal: Path):
        """Series interleave line by line, so striding the file would keep one
        key and drop the other entirely."""
        records = list(
            mp.io.mlp_jsonl.MlpJsonlMetricReader().read(
                wal, source="x", request=Request(stride=3)
            )
        )
        assert {r["k"] for r in records} == {"energy", "temp"}
        assert len(records) == 4  # two samples of each of the two series

    def test_a_torn_last_line_is_skipped(self, tmp_path: Path):
        """A live writer can be caught mid-line: the unterminated tail is not a
        record yet, and the next poll reads it whole."""
        path = tmp_path / "metrics.mlp.jsonl"
        path.write_text('{"t":"scalar","k":"a","v":1}\n{"t":"sca', encoding="utf-8")
        records = list(mp.io.mlp_jsonl.MlpJsonlMetricReader().read(path, source="x"))
        assert [r["v"] for r in records] == [1]

    def test_a_complete_last_line_without_newline_is_a_record(self, tmp_path: Path):
        path = tmp_path / "metrics.mlp.jsonl"
        path.write_text('{"k":"a","v":1}\n{"k":"a","v":2}', encoding="utf-8")
        assert [
            r["v"]
            for r in mp.io.mlp_jsonl.MlpJsonlMetricReader().read(path, source="x")
        ] == [
            1,
            2,
        ]

    @pytest.mark.parametrize(
        "bad",
        ["not json", "[1, 2]", '{"t": "scalar", "v": 1}'],
        ids=["not-json", "not-an-object", "no-key"],
    )
    def test_a_malformed_line_before_the_end_raises(self, tmp_path: Path, bad: str):
        """Only a torn tail is forgiven: a bad line anywhere else is corruption,
        and skipping it would shift every later record's resume index."""
        path = tmp_path / "metrics.mlp.jsonl"
        path.write_text(
            f'{{"k":"a","v":1}}\n{bad}\n{{"k":"a","v":2}}\n', encoding="utf-8"
        )
        with pytest.raises(ValueError, match=r"metrics\.mlp\.jsonl:2"):
            list(mp.io.mlp_jsonl.MlpJsonlMetricReader().read(path, source="x"))

    def test_a_malformed_terminated_last_line_raises(self, tmp_path: Path):
        """A last line that ends in a newline was finished by its writer."""
        path = tmp_path / "metrics.mlp.jsonl"
        path.write_text('{"k":"a","v":1}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"metrics\.mlp\.jsonl:2"):
            list(mp.io.mlp_jsonl.MlpJsonlMetricReader().read(path, source="x"))

    def test_blank_lines_are_not_records(self, tmp_path: Path):
        path = tmp_path / "metrics.mlp.jsonl"
        path.write_text('\n{"k":"a","v":1}\n\n{"k":"a","v":2}\n', encoding="utf-8")
        records = mp.io.mlp_jsonl.MlpJsonlMetricReader().read(
            path, source="x", request=Request(since=1)
        )
        assert [r["v"] for r in records] == [2]

    def test_the_source_tag_never_overrides_the_writers(self, tmp_path: Path):
        path = tmp_path / "metrics.mlp.jsonl"
        path.write_text(
            '{"k":"a","v":1,"tags":{"source":"mine","run":3}}\n', encoding="utf-8"
        )
        (record,) = mp.io.mlp_jsonl.MlpJsonlMetricReader().read(path, source="x")
        assert record["tags"] == {"source": "mine", "run": 3}


def write_mrec(path: Path, steps, times=None) -> Path:
    frame = mp.Frame()
    frame["atoms"] = {"x": [0.0], "y": [0.0], "z": [0.0]}
    mp.io.write_mrec(
        path,
        trajectory=mp.Trajectory(
            [frame] * len(steps),
            step=np.array(steps),
            time=None if times is None else np.array(times),
        ),
    )
    return path


class TestMrecReader:
    def test_claims_a_record_directory(self, tmp_path: Path):
        path = write_mrec(tmp_path / "growth.mrec", [0, 10])
        assert mp.io.mrec.MrecMetricReader().sniff(path)
        assert not mp.io.mrec.MrecMetricReader().sniff(tmp_path / "growth.xyz")

    def test_claims_a_packed_record(self, tmp_path: Path):
        """molrs packs a closed record into ``*.mrec.zip``; it is the same run."""
        packed = Path(
            mp.io.mrec.pack_mrec_zip(str(write_mrec(tmp_path / "run.mrec", [0, 10])))
        )
        assert packed.name == "run.mrec.zip"
        assert "**/*.mrec.zip" in mp.io.mrec.MrecMetricReader.patterns
        assert mp.io.mrec.MrecMetricReader().sniff(packed)
        records = list(mp.io.mrec.MrecMetricReader().read(packed, source="run"))
        assert [r["v"] for r in records if r["k"] == "mrec/step"] == [0.0, 10.0]

    def test_rejects_what_only_looks_like_a_record(self, tmp_path: Path):
        empty = tmp_path / "empty.mrec"
        empty.mkdir()
        junk = tmp_path / "junk.mrec.zip"
        junk.write_bytes(b"not a zip")
        plain = tmp_path / "plain.zip"
        plain.write_bytes(b"PK")
        assert not mp.io.mrec.MrecMetricReader().sniff(empty)
        assert not mp.io.mrec.MrecMetricReader().sniff(junk)
        assert not mp.io.mrec.MrecMetricReader().sniff(plain)

    def test_sniff_and_read_close_the_store(self, tmp_path: Path, monkeypatch):
        module = mp.io.mrec
        native = mp.io.mrec.MrecReader  # bound before the patch below replaces it

        opened: list[Tracked] = []

        class Tracked:
            """Delegates to the native cursor and records that it was closed."""

            def __init__(self, path):
                self._inner = native(path)
                self.closed = False
                opened.append(self)

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *exc):
                self.closed = True
                return self._inner.__exit__(*exc)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        monkeypatch.setattr(module, "MrecReader", Tracked)
        path = write_mrec(tmp_path / "run.mrec", [0, 10])
        assert mp.io.mrec.MrecMetricReader().sniff(path)
        list(
            mp.io.mrec.MrecMetricReader().read(
                path, source="run", request=Request(limit=1)
            )
        )
        assert len(opened) == 2
        assert all(reader.closed for reader in opened)

    def test_emits_the_step_and_time_series(self, tmp_path: Path):
        path = write_mrec(tmp_path / "run.mrec", [0, 10, 20], [0.0, 0.5, 1.0])
        records = list(mp.io.mrec.MrecMetricReader().read(path, source="run"))
        steps = [r["v"] for r in records if r["k"] == "mrec/step"]
        times = [r["v"] for r in records if r["k"] == "mrec/time"]
        assert steps == [0.0, 10.0, 20.0]
        assert times == [0.0, 0.5, 1.0]
        assert all(r["tags"] == {"source": "run"} for r in records)

    def test_s_is_the_simulation_step_not_the_frame_index(self, tmp_path: Path):
        path = write_mrec(tmp_path / "run.mrec", [0, 10, 20], [0.0, 0.5, 1.0])
        records = list(mp.io.mrec.MrecMetricReader().read(path, source="run"))
        times = [r for r in records if r["k"] == "mrec/time"]
        assert [r["s"] for r in times] == [0.0, 10.0, 20.0]
        strided = mp.io.mrec.MrecMetricReader().read(
            path, source="run", request=Request(stride=2)
        )
        assert [r["s"] for r in strided if r["k"] == "mrec/time"] == [0.0, 20.0]
