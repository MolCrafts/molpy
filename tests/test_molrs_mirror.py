"""molpy.md is a verbatim re-export of molrs.md."""

from __future__ import annotations

import molpy.md
import molrs.md


def test_every_molrs_md_name_is_the_same_object() -> None:
    for name in molrs.md.__all__:
        assert getattr(molpy.md, name) is getattr(molrs.md, name)


def test_molpy_md_adds_no_extra_public_names() -> None:
    extra = set(molpy.md.__all__) - set(molrs.md.__all__)
    assert not extra, extra
