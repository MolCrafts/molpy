from molpy.core.entity import Entity, Link


class TestLink:
    def test_endpoints_and_replace(self) -> None:
        a = Entity({"id": 1})
        b = Entity({"id": 2})
        c = Entity({"id": 3})
        link = Link([a, b], order=1)
        assert link.endpoints == (a, b)
        assert link["order"] == 1
        link.replace_endpoint(b, c)
        assert link.endpoints == (a, c)
