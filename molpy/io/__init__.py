import molpy.io.lammps as lammps

Readers = {
    'DataReaders': {
        'lammps': lammps.DataReader,
    },
    'TrajReaders': {
        'lammps': lammps.TrajReader,
    }
}