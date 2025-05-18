import molpy as mp

class ConvertForceField:
    
    def __init__(self, url: str):
        self.url = url

    def fetch(self):
        import requests
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch data from {self.url}. Status code: {response.status_code}")
        
    def parse(self): ...


class ConvertUFF(ConvertForceField):
    
    def __init__(self):
        super().__init__("https://github.com/openbabel/openbabel/blob/master/data/UFF.prm")
        self.ff = mp.ForceField("UFF")
        

    def parse(self):

        atomstyle = self.ff.def_atomstyle("full")
        bondstyle = self.ff.def_bondstyle("harmonic")
        anglestyle = self.ff.def_anglestyle("cosine")
        dihedralstyle = self.ff.def_dihedralstyle("cosine")

        f = self.fetch()
        lines = f.split("\n")

        for line in lines:

            if line.startswith("#"):
                continue
            if line.startswith("atom"):
                self.parse_atom(line)

    def parse_atom(self, line: str):

        _, smart, type_, descr = line.split()
        