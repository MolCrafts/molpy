"""Tests for base Adapter class."""

import pytest

from molpy.adapter.base import Adapter


class MockAdapter(Adapter[str, int]):
    """Mock adapter for testing base class functionality."""

    def _do_sync_to_internal(self) -> None:
        if self._external is not None:
            self._internal = str(self._external)

    def _do_sync_to_external(self) -> None:
        if self._internal is not None:
            self._external = int(self._internal) if self._internal.isdigit() else None


class TestAdapter:
    """Tests for base Adapter class."""

    def test_init_with_internal(self):
        """Test initialization with internal representation."""
        adapter = MockAdapter(internal="123")
        assert adapter.has_internal()
        assert not adapter.has_external()
        assert adapter.get_internal() == "123"

    def test_init_with_external(self):
        """Test initialization with external representation."""
        adapter = MockAdapter(external=456)
        assert not adapter.has_internal()
        assert adapter.has_external()
        assert adapter.get_external() == 456

    def test_init_with_both(self):
        """Test initialization with both representations."""
        adapter = MockAdapter(internal="123", external=456)
        assert adapter.has_internal()
        assert adapter.has_external()
        assert adapter.get_internal() == "123"
        assert adapter.get_external() == 456

    def test_get_internal_syncs_from_external(self):
        """Test that get_internal syncs from external if needed."""
        adapter = MockAdapter(external=789)
        internal = adapter.get_internal()
        assert internal == "789"
        assert adapter.has_internal()

    def test_get_external_syncs_from_internal(self):
        """Test that get_external syncs from internal if needed."""
        adapter = MockAdapter(internal="456")
        external = adapter.get_external()
        assert external == 456
        assert adapter.has_external()

    def test_get_internal_raises_when_none(self):
        """Test that get_internal raises when both are None."""
        adapter = MockAdapter()
        with pytest.raises(ValueError, match="both internal and external are None"):
            adapter.get_internal()

    def test_get_external_raises_when_none(self):
        """Test that get_external raises when both are None."""
        adapter = MockAdapter()
        with pytest.raises(ValueError, match="both internal and external are None"):
            adapter.get_external()

    def test_set_internal(self):
        """Test setting internal representation."""
        adapter = MockAdapter()
        adapter.set_internal("999")
        assert adapter.get_internal() == "999"

    def test_set_external(self):
        """Test setting external representation."""
        adapter = MockAdapter()
        adapter.set_external(888)
        assert adapter.get_external() == 888

    def test_sync_to_internal_raises_when_no_external(self):
        """Test that sync_to_internal raises when external is None."""
        adapter = MockAdapter()
        with pytest.raises(ValueError, match="external representation is None"):
            adapter.sync_to_internal()

    def test_sync_to_external_raises_when_no_internal(self):
        """Test that sync_to_external raises when internal is None."""
        adapter = MockAdapter()
        with pytest.raises(ValueError, match="internal representation is None"):
            adapter.sync_to_external()

    def test_check_passes_with_internal(self):
        """Test that check passes when internal is set."""
        adapter = MockAdapter(internal="123")
        adapter.check()  # Should not raise

    def test_check_passes_with_external(self):
        """Test that check passes when external is set."""
        adapter = MockAdapter(external=456)
        adapter.check()  # Should not raise

    def test_check_raises_when_neither_set(self):
        """Test that check raises when neither is set."""
        adapter = MockAdapter()
        with pytest.raises(ValueError, match="neither internal nor external"):
            adapter.check()

    def test_repr(self):
        """Test string representation."""
        adapter1 = MockAdapter(internal="123")
        assert "internal=set" in repr(adapter1)
        assert "external=None" in repr(adapter1)

        adapter2 = MockAdapter(external=456)
        assert "internal=None" in repr(adapter2)
        assert "external=set" in repr(adapter2)

        adapter3 = MockAdapter(internal="123", external=456)
        assert "internal=set" in repr(adapter3)
        assert "external=set" in repr(adapter3)
