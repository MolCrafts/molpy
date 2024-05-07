
class Plugins:

    plugins = []

    @classmethod
    def register(self, plugin):
        self.plugins.append(plugin)
        plugin.when_register()

    def do(self, stage:str):
        for plugin in self.plugins:
            if hasattr(plugin, stage):
                getattr(plugin, stage)()
    
def register_plugin(plugin):
    Plugins.register(plugin)

class Plugin:

    def __init__(self, name):
        self.name = name
        Plugins.plugins.append(self)

    def __str__(self):
        return self.name
    
    def when_register(self):
        pass