import string
import textwrap

class Script:

    def __init__(self):
        self._text:list[str] = []

    @classmethod
    def from_text(cls, text:str):
        script = cls()
        script.text = text.split("\n")
        return script
    
    @classmethod
    def from_file(cls, path:str):
        with open(path, "r") as f:
            return cls.from_text(f.read())
        
    @property
    def text(self):
        text = "\n".join(self._text)
        text = Script.format(text)
        return text

    def append(self, text:str):
        self._text.append(text)

    def insert(self, index:int, text:str):
        self._text.insert(index, text)

    def write(self, path:str):
        with open(path, "w") as f:
            f.write("\n".join(self.text))

    def substitute(self, mapping:dict[str,str]):
        for i in range(len(self._text)):
            self._text[i] = string.Template(self._text[i]).safe_substitute(mapping)

    @staticmethod
    def format(text:str):
        return textwrap.dedent(text).strip()
