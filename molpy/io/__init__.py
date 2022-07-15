import molpy.io.lammps as lammps
from logging import Logger

ioLog = Logger('molpy.io')

Readers = {
    'DataReaders': {
        'lammps': lammps.DataReader,
    },
    'TrajReaders': {
        'lammps': lammps.DumpReader,
    }
}