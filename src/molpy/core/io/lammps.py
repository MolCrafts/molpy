# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from pathlib import Path
from ..forcefield import ForceField

def scroll_down(handler, n=1) -> str:
    for _ in range(n):
        line = handler.readline()
    return line


class LAMMPSLog:
    def __init__(self, fpath, mode="r"):
        self.fpath = Path(fpath)

        self.stage_info = {}
        self.stage_summary = {}

        self._summary = {}

    def read(self):
        with open(self.fpath, "r") as f:
            stage_index = 0
            stage_style = "md"
            while line := f.readline():
                if line.startswith("minimize"):
                    stage_style = "minimize"

                if line.startswith("Per MPI rank memory allocation"):
                    headers = next(f).split()
                    stage_info = self._read_stage(f, headers)
                    if stage_style == "minimize":
                        scroll_down(f, 3)
                        stage_summary = self._read_min_stats(f)
                        stage_style = "md"
                    else:
                        scroll_down(f, 1)
                        stage_summary = self._read_md_stats(f)
                    self.stage_info[stage_index] = stage_info
                    self.stage_summary[stage_index] = stage_summary
                    stage_index += 1

                if line.startswith("Total wall time"):
                    hours, minutes, seconds = map(int, line.split(":")[1:])
                    self._summary["total_time"] = hours * 3600 + minutes * 60 + seconds
                    continue

    def _path_check(self, fpath: Path):
        if not fpath.exists():
            raise FileNotFoundError(f"{fpath} does not exist.")

    def _read_min_stats(self, handler):
        stage_summary = {}
        line = handler.readline()
        assert line.startswith("Minimization stats:")
        # TODO: read minimization stats

        return stage_summary

    def _read_md_stats(self, handler):
        stage_summary = {}
        line = handler.readline()
        assert line.startswith("Performance:")
        values = line.split()
        stage_summary["n_ns_per_day"] = float(values[1])
        stage_summary["n_h_per_ns"] = float(values[3])
        stage_summary["n_ts_per_s"] = float(values[5])

        return stage_summary

    def _read_stage(self, handler, headers):
        stage_info = {header: [] for header in headers}
        while line := handler.readline():
            if line.startswith("Loop time of"):
                return stage_info
            values = line.split()
            for header, value in zip(stage_info.keys(), values):
                stage_info[header].append(value)


class LammpsScriptParser:

    def __init__(self, fpath:str | Path, mode='r'):
        self._fpath = Path(fpath)
        self._ff = {}

        self._file = open(self._fpath, mode)

    def __del__(self):
        self._file.close()

    def parse(self):
        
        for line in self._file.readlines():

            line = line.strip()

            if line.startswith('#'):
                continue

            if line.startswith('pair'):
                self.parse_pair(line)

    def parse_pair(self, line:str):

        pair = self._ff['pair']

        if line.startswith('pair_style'):
            cmd, style, *global_args = line.split()
            assert cmd == 'pair_style'

            pair[style] = {
                'global_args': global_args,
                'pair_coeffs': {},
            }

        elif line.startswith('pair_coeff'):

            cmd, *args = line.split()
            assert cmd == 'pair_coeff'
            if not 'hybrid' in args:
                assert pair.keys() == 1
                style = pair.keys()[0]
                type1, type2 = args[:2]
                pair[style]['pair_coeffs'][type1] = {type2: [float(arg) for arg in args[2:]]}

        elif line.startswith('pair_modift'):

            cmd, *args = line.split()
            assert cmd == 'pair_modify'
            
        else:
            pass
