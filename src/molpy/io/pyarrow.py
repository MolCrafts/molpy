"""
PyArrow conversion utilities for molpy Frame and DataFrame objects.
"""
import molpy as mp


def to_arrow(
    frame: mp.Frame,
    atom_fields: list[str] = ["name", "element"],
    bond_fields: list[str] = ["i", "j", "order"],
) -> dict:
    """
    Convert atoms and bonds from a molpy Frame to pyarrow IPC buffers.
    Returns a dict: { 'atoms': <arrow_buffer>, 'bonds': <arrow_buffer> or None }
    """
    from pyarrow import Table, BufferOutputStream, ipc

    result = {}
    # Atoms
    atom_fields = ["xyz", *atom_fields]
    atoms = frame["atoms"][atom_fields]
    # Always use .to_dataframe().reset_index() for xarray.Dataset
    atoms_df = atoms.to_dataframe().reset_index()
    atoms_arrow = Table.from_pandas(atoms_df)
    sink = BufferOutputStream()
    with ipc.new_stream(sink, atoms_arrow.schema) as writer:
        writer.write_table(atoms_arrow)
    result["atoms"] = sink.getvalue()
    # Bonds (optional)
    bonds = frame.get("bonds", None)
    if bonds is not None:
        bonds = bonds[bond_fields]
        bonds_df = bonds.to_dataframe().reset_index()
        bonds_arrow = Table.from_pandas(bonds_df)
        sink = BufferOutputStream()
        with ipc.new_stream(sink, bonds_arrow.schema) as writer:
            writer.write_table(bonds_arrow)
        result["bonds"] = sink.getvalue()
    else:
        result["bonds"] = None
    print(result)
    return result
