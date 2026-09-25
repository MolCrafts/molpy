"""``molpy.builder`` facade — placement vocabulary is molrs-owned."""

from __future__ import annotations

import molrs
import pytest

import molpy.builder as builder


@pytest.mark.parametrize("name", ["Trace", "Orienter", "LineOrienter", "TangOrienter"])
def test_placement_vocabulary_is_the_native_class(name: str) -> None:
    assert getattr(builder, name) is getattr(molrs, name)
    assert name in builder.__all__
