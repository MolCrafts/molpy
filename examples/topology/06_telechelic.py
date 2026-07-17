"""T10 — Telechelic oligomer: CAPA–(EO)n–CAPB.

Head cap carries only SITE a; tail cap only SITE b so both ends of the path
can form an ether bond with EO.

Guide: docs/user-guide/topology/06_telechelic.md
Run:   python topology/06_telechelic.py
"""

from molpy.core import fields

from eo_kit import eo_builder, full_library, report


def main() -> None:
    lib = full_library()
    builder = eo_builder(extra={"CAPA": lib["CAPA"], "CAPB": lib["CAPB"]})
    tele = builder.build_sequence(["CAPA"] + ["EO"] * 6 + ["CAPB"])
    report("telechelic", tele)
    by_id = {int(a[fields.RES_ID]): str(a[fields.RES_NAME]) for a in tele.atoms}
    seq = [by_id[i] for i in sorted(by_id)]
    print(f"  sequence: {'-'.join(seq)}")


if __name__ == "__main__":
    main()
