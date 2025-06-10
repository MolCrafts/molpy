import operator
from functools import reduce
from typing import Any, Iterator, Callable, Sequence, Union
import numpy as np
from .arraydict import ArrayDict

NestedKey = str | list[str]  # type_check_only
dictlike = Union[dict, "NestDict"]

from collections.abc import MutableMapping


class NestDict(MutableMapping):

    def __init__(self, source: dict = {}):
        """Initializes a NestDict with or without a source dictionary.

        Args:
            source (dict, optional): Source dictionary. Defaults to None.

        Raises:
            TypeError: source must be a dict

        Examples:

        >>> nd = NestDict()
        >>> len(nd)
        0
        >>> NestDict({"a": 1, "b": 2}).keys()
        dict_keys(['a', 'b'])
        >>> nd = NestDict({'a': {'b': {'c': 1}}})
        >>> nd[['a', 'b', 'c']]
        1
        >>> nd.get('a.b.c')
        1
        """
        super().__setattr__("_data", dict(source))

    @classmethod
    def _construct(cls, data, nested_key: NestedKey):
        """_construct_path is a recursive function that constructs a nested path in a dictionary.

        Args:
            data (dict): The dictionary to construct the path in.
            nested_key (NestedKey): The nested path to construct.

        Returns:
            dict: The constructed dictionary.
        """
        if nested_key:
            key = nested_key[0]
            if key not in data:
                data[key] = cls()
            return NestDict._construct(data[key], nested_key[1:])

    def _traverse(self, nested_key: NestedKey, construct: bool = False) -> Any:
        """Traverses a nested path in a dictionary.

        Args:
            nested_key (NestedKey): The nested path to traverse.
            construct (bool, optional): if create new container . Defaults to False.

        Raises:
            KeyError: if nested_key is not found.

        Returns:
            Any: The value at the end of the nested path.
        """
        if construct:
            self._construct(self._data, nested_key)

        if isinstance(nested_key, list):
            return reduce(operator.getitem, nested_key, self._data)
        return self._data[nested_key]

    def __bool__(self) -> bool:
        return bool(self._data)

    def __eq__(self, other: Any) -> bool:
        return self._data == other

    def flatten(self) -> dict:
        """get a python dict with a flat structure. The key is the nested key joined by the separator.

        Args:
            separator (str, optional): separator of nested key. Defaults to ".".

        Returns:
            dict: a python dict with a flat structure.

        Examples:
            >>> NestDict({'a': {'b': {'c': 1}}}).flatten()
            {('a', 'b', 'c'): 1}
        """

        def _flatten(data, parent_key: list[str] = []):
            items = []
            for k, v in data.items():
                new_key = [*parent_key, k]
                if isinstance(v, MutableMapping):
                    items.extend(_flatten(v, new_key))
                else:
                    items.append((tuple(new_key), v))
            return items

        flat_dict = dict(_flatten(self._data, []))
        return flat_dict

    def __getitem__(self, nested_key: NestedKey) -> Any:
        item = self._traverse(nested_key)
        return item

    def __delitem__(self, nested_key: NestedKey) -> None:
        if isinstance(nested_key, list):
            parent = self._traverse(nested_key[:-1])
            del parent[nested_key[-1]]
        else:
            del self._data[nested_key]

    def __setitem__(self, nested_key: NestedKey, value: Any) -> None:
        if isinstance(nested_key, list):
            dest_key = nested_key[-1]
            parent = self._traverse(nested_key[:-1], construct=True)
            parent[dest_key] = value
        else:
            self._data[nested_key] = value

    def get(self, nested_path: str, default, sep: str = "."):
        """get a value from a nested path in a dictionary.

        Args:
            nested_path (str): The nested path to traverse.
            sep (str, optional): Defaults to '.'.

        Returns:
            Any: The value at the end of the nested path.

        Examples:
            >>> nd = NestDict({'a': {'b': {'c': 1}}})
            >>> nd.get('a.b.c')
            1

        """
        path = nested_path.split(sep)
        try:
            return self._traverse(path)
        except KeyError:
            return default

    def __contains__(self, nested_key: NestedKey) -> bool:
        try:
            self._traverse(nested_key)
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __ne__(self, other: Any) -> bool:
        return self._data != other

    def set(self, nested_path: str, value: Any, sep: str = ".") -> None:
        """set a value at a nested path in a dictionary.

        Args:
            nested_path (str): The nested path to traverse.
            value (Any): The value to set.
            sep (str, optional): Defaults to '.'.

        Examples:
            >>> nd = NestDict()
            >>> nd.set('a.b.c', 1)
            >>> nd['a']['b']['c']
            1
            >>> nd.set('a.b.c', 2)
            >>> nd[['a', 'b', 'c']]
            2
        """
        path = nested_path.split(sep)
        self._traverse(path[:-1], construct=True)[path[-1]] = value

    def __str__(self) -> str:
        return f"<{str(self._data)}>"

    def __repr__(self) -> str:
        return f"<{repr(self._data)}>"

    def clear(self) -> None:
        return self._data.clear()

    def copy(self) -> dict:
        return self._data.copy()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def update(self, other: dict):
        self._data.update(other)

    def concat(self, others: Sequence["NestDict"], fallback: dict[type, Callable] = {}):
        """Concatenate two dictionaries with compatible types."""
        if not isinstance(others, Sequence):
            others = [others]
        if not all(isinstance(other, NestDict) for other in others):
            raise TypeError(f"Cannot concatenate `{type(others)}` to NestDict.")

        for other in others:
            for k, v in other.items():
                if k not in self._data:
                    raise KeyError(f"Key '{k}' not in self")

                this = self._data[k]

                if type(this) is not type(v):
                    # try:
                    #     self._data[k] = fallback[type(this)](this, v)
                    # except:
                    #     raise TypeError(
                    #         f"Type mismatch for key '{k}': "
                    #         f"expected {type(this)}, got {type(v)}"
                    #     )
                    self._data[k] = fallback[type(v)](this, v)

                elif isinstance(this, NestDict):
                    this.concat(v)

                elif isinstance(this, dict):
                    self._data[k].update(v)

                elif isinstance(v, list):
                    self._data[k] = this + v

                elif isinstance(v, np.ndarray):
                    self._data[k] = np.concatenate((this, v))

                elif isinstance(v, ArrayDict):
                    self._data[k].concat(v)

                else:
                    self._data[k] = fallback[type(this)](this, v)
        return self

    def __add__(self, other: dictlike) -> "NestDict":
        """Concatenate two dictionaries with compatible types."""
        if not isinstance(other, dictlike):
            raise TypeError(f"Cannot concatenate `{type(other)}` to NestDict.")
        result = self.copy()
        result.concat(other)
        return result

    def __copy__(self) -> "NestDict":
        """Return a shallow copy of the NestDict."""
        return NestDict({k: v for k, v in self._data.items()})
    
    def to_hdf5(self, h5file):
        """Save the NestDict to an HDF5 file.

        Args:
            h5file (h5py.File): The HDF5 file to save to.
        """
        def _to_hdf5(data, h5group):
            for k, v in data.items():
                if hasattr(v, "to_hdf5"):
                    group = h5group.create_group(str(k))
                    v.to_hdf5(group)
                elif isinstance(v, dict):
                    group = h5group.create_group(str(k))
                    _to_hdf5(v, group)
                else:
                    h5group.create_dataset(str(k), data=v)
        _to_hdf5(self._data, h5file)
        return h5file
            
            
def concat(nds: Sequence[dictlike]):
    """Concatenate a list of dictionaries or NestDicts.

    Args:
        nds (Sequence[dictlike]): A sequence of dictionaries or NestDicts to concatenate.

    Returns:
        NestDict: A new NestDict containing the concatenated data.
    """
    result = nds[0].copy()
    for nd in nds[1:]:
        result.concat(nd)
    return result