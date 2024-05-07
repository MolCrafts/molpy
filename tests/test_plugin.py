import pytest
import molpy as mp

class TestPlugins:

    @pytest.fixture(scope="class", name="plugin")
    def init_plugin(self):

        class TestPlugin(mp.Plugin):
            def __init__(self):
                super().__init__("TestPlugin")
            
            def when_register(self):
                print("TestPlugin registered")
                def test(self, greeting):
                    self['test'] = greeting
                mp.Struct.test = test

        return TestPlugin()

    def test_init(self, plugin):

        mp.register_plugin(plugin)
        struct = mp.StaticStruct()
        struct.test("hello")
        assert struct['test'] == "hello"
