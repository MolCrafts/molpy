from molpy._alias import AliasSystem, NameSpace, AliasEntry
import pytest

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            return cls._instances[cls]
        else:
            return cls._instances[cls](*args, **kwargs)

class TestAlias:

    @pytest.fixture(scope="function", name="Alias")
    def alias(self):
        Alias = AliasSystem.reset()
        Alias.new_namespace("default")
        return Alias

    def test_set_get_attr(self, Alias):

        # if alias not in system, return alias as key
        assert Alias.not_exist_key == "not_exist_key"

        # if alias in system, return key
        Alias.set("key_in_default", "key_in_default", int, None, "")
        assert Alias.key_in_default == "key_in_default"

        # if get namespace, return namespace
        assert isinstance(Alias.default, NameSpace)
        assert Alias.default.name == "default"

        # if method, return method
        assert Alias.list() == {"default": ["not_exist_key", "key_in_default"]}

    def test_set_get_entry(self, Alias):

        # if alias not in system, raise KeyError
        with pytest.raises(KeyError):
            Alias["not_exist_key"]

        # if alias in system, return AliasEntry
        Alias.set("key_in_default", "key_in_default", int, None, "")
        assert isinstance(Alias["key_in_default"], AliasEntry)
        assert Alias["key_in_default"].type == int

        # if get namespace, return namespace
        assert isinstance(Alias["default"], NameSpace)
        assert Alias.default.name == "default"

        # then, unit and type can be accessed
        assert Alias["key_in_default"].unit == None
        assert Alias["key_in_default"].type == int
