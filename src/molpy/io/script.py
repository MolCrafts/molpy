from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile

class Script(Path):

    def __init__(self, fpath: Path, temporary: bool = False):
        self.temporary = temporary
        if self.temporary:
            fp = NamedTemporaryFile(prefix=fpath.stem, delete_on_close=True, mode='w+')
        else:
            fp = open(fpath, 'w+')
        self._fp = fp
        super().__init__(fpath)

    def __del__(self):
        self._fp.close()

    def __repr__(self):
        return f'<Script: {self.name}>'
    
    def __str__(self):
        return self.text
    
    def read(self) -> str:
        return self._fp.read()
    
    def write(self, text: str):
        self._fp.write(text)

    def substitute(self, mapping: dict):
        text = self.read()
        template = Template(text)
        text = template.substitute(mapping)
        self.write(text)
    
    @property
    def text(self):
        return self.read()