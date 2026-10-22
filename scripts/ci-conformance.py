#!/usr/bin/env python3
"""Run molrec's conformance suite against the Rust core and write its snapshot.

The suite enters through the ``*.mrec`` doors at the top of ``molpy.io``
(``read_mrec_frame`` / ``write_mrec_frame``, ``read_mrec_system`` /
``write_mrec_system``, ``read_mrec_trajectory``, ``read_mrec_meta``, …) and
the store classes of
``molpy.io.mrec`` (``MrecReader``, ``MrecWriter``, ``SequenceSchema``,
``ForceFieldSection``), all implemented in ``crates/molrs/src/io/mrec``. This does not
collect ``molpy/tests`` or any other binding suite. An adapter author writes two methods; every assertion stays
in molrec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import molci as mci
from molcrafts_ci.producers import detect_profile, github_source
from molrec.report import CaseResult, Report
from molrec.suite import ConformanceSuite

_DETAIL_LIMIT = 500


def _detail(result: CaseResult) -> str:
    text = result.message or "; ".join(str(v) for v in result.violations)
    if len(text) <= _DETAIL_LIMIT:
        return text
    return text[: _DETAIL_LIMIT - 1] + "…"


def _payload(report: Report) -> dict:
    passed = failed = skipped = 0
    failures: list[dict[str, str]] = []
    for result in report.results:
        if result.status == "pass":
            passed += 1
        elif result.status == "skip":
            skipped += 1
        else:
            failed += 1
            failures.append(
                {
                    "case_id": result.case_id,
                    "module": result.module,
                    "backend": result.backend,
                    "direction": result.direction,
                    "status": result.status,
                    "detail": _detail(result),
                }
            )

    # A suite that collected nothing is a red run. Publishing passed=0,
    # failed=0 would read as a clean empty record.
    if not report.results:
        failed = 1
        failures.append(
            {
                "case_id": "*",
                "module": "",
                "backend": "",
                "direction": "",
                "status": "error",
                "detail": "suite ran nothing",
            }
        )

    modules = sorted({result.module for result in report.results if result.module})
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "implementation": report.implementation,
        "version": report.version,
        "modules": ", ".join(modules),
        "failures": failures,
    }


def _snapshot(payload: dict, *, track: bool) -> mci.Snapshot:
    return mci.Snapshot(
        manifest=mci.Manifest(
            record="conformance",
            source=github_source(require_commit=track),
            producer="molrec",
            profile=detect_profile(),
            tracking=mci.Tracking(enabled=track, generation=1),
        ),
        payload=payload,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="molrec's tests directory, which holds molrs_adapter.py",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Snapshot JSON to write"
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Mark the snapshot as project history. A pull request must not.",
    )
    args = parser.parse_args(argv)

    suite = args.suite.resolve()
    if not (suite / "molrs_adapter.py").is_file():
        print(f"no molrs adapter at {suite / 'molrs_adapter.py'}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(suite))
    import molrs_adapter

    implementation = molrs_adapter.Molrs()
    modules = sorted(implementation.adapters())
    report = ConformanceSuite(implementation, modules=modules).run()
    print(report.table())

    try:
        snapshot = _snapshot(_payload(report), track=args.track)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({snapshot.snapshot_id()})")
    return 0 if report.ok and report.results else 1


if __name__ == "__main__":
    raise SystemExit(main())
