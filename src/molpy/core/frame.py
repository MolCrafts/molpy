import numpy as np
import xarray as xr
from collections.abc import MutableMapping
from typing import Any, Sequence, Union, Dict, Optional

from .box import Box
from .forcefield import ForceField


def _dict_to_dataset(data: Dict[str, Any]) -> xr.Dataset:
    """Convert a mapping of arrays to an xarray.Dataset.
    
    Each key becomes a DataArray in the Dataset, with proper dimensions.
    All arrays are assumed to be aligned on the 'index' dimension.
    """
    if not data:
        # 返回空Dataset，但包含基本的index坐标
        return xr.Dataset(coords={"index": np.array([], dtype=int)})
    
    data_vars = {}
    coords = {}
    
    # First pass: determine the index size and create coordinates
    index_size = None
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim >= 1:
            if index_size is None:
                index_size = arr.shape[0]
            elif arr.shape[0] != index_size:
                raise ValueError(f"All arrays must have the same first dimension size. "
                               f"Got {arr.shape[0]} for '{k}', expected {index_size}")
    
    if index_size is None:
        index_size = 0
    
    coords["index"] = np.arange(index_size)
    
    # Second pass: create DataArrays with proper dimensions
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim == 0:
            # Scalar - broadcast to all indices
            data_vars[k] = ("index", np.full(index_size, arr.item()))
        elif arr.ndim == 1:
            # 1D array
            data_vars[k] = ("index", arr)
        elif arr.ndim == 2:
            if arr.shape[1] == 3:  # Assume spatial coordinates
                data_vars[k] = (("index", "spatial"), arr)
                if "spatial" not in coords:
                    coords["spatial"] = ["x", "y", "z"]
            else:
                # Other 2D arrays
                data_vars[k] = (("index", f"{k}_dim"), arr)
                coords[f"{k}_dim"] = np.arange(arr.shape[1])
        else:
            # Higher dimensional arrays - flatten extra dimensions
            new_shape = (arr.shape[0], -1)
            reshaped = arr.reshape(new_shape)
            data_vars[k] = (("index", f"{k}_dim"), reshaped)
            coords[f"{k}_dim"] = np.arange(reshaped.shape[1])
    
    return xr.Dataset(data_vars, coords=coords)


# _dict_to_dataarray 已废弃，兼容性导出一个空壳防止import错误
def _dict_to_dataarray(*args, **kwargs):
    raise ImportError("_dict_to_dataarray is deprecated. Use _dict_to_dataset instead.")


class Frame(MutableMapping):
    """Container of simulation data based on :class:`xarray.Dataset`."""
    def __init__(self, data: Optional[Dict[str, Union[Dict[str, Any], xr.Dataset]]] = None, 
                 *, box: Optional[Box] = None, forcefield: Optional[ForceField] = None, meta: Optional[Dict[str, Any]] = None, **extra_meta):
        """Initialize Frame.
        
        Parameters
        ----------
        data : dict, optional
            Dictionary mapping keys to either:
            - Dict[str, np.ndarray]: Dictionary of arrays/scalars. Non-scalar arrays must have the same first dimension length.
            - xr.Dataset: Pre-built Dataset
        box : Box, optional
            Simulation box
        """
        self._data: Dict[str, xr.Dataset] = {}
        self.box = box
        self._meta: Dict[str, Any] = {}
        if meta is not None:
            self._meta.update(meta)
        self._meta.update(extra_meta)
        
        if data:
            for key, value in data.items():
                self[key] = value # Calls __setitem__

    # Mapping protocol
    def __getitem__(self, key: str) -> xr.Dataset:
        if key in self._data:
            return self._data[key]
        if key in self._meta:
            return self._meta[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "box":
            if value is not None and not isinstance(value, Box):
                raise TypeError("Box must be a Box instance or None")
            self.box = value
            return
        # meta keys (e.g. timestep, etc.)
        if key in self._meta or np.isscalar(value):
            self._meta[key] = value
            return
        if isinstance(value, xr.Dataset):
            self._data[key] = value
        elif isinstance(value, dict):
            self._data[key] = _dict_to_dataset(value)
        else:
            raise TypeError(f"Frame values for key '{key}' must be xarray.Dataset, dict (for _dict_to_dataset), or scalar. Got {type(value)}")

    def __delitem__(self, key: str) -> None:
        if key == "box":
            self.box = None
        elif key in self._meta:
            del self._meta[key]
        elif key in self._data:
            del self._data[key]
        else:
            raise KeyError(key)

    def __iter__(self):
        yield from self._data.keys()
        yield from self._meta.keys()
        if self.box is not None:
            yield "box"
        if self.timestep is not None:
            yield "timestep"

    def __len__(self) -> int:
        n = len(self._data) + len(self._meta)
        if self.box is not None:
            n += 1
        return n

    def __repr__(self) -> str:
        content_keys = list(self._data.keys())
        meta_keys = list(self._meta.keys())
        special_keys = []
        if self.box is not None:
            special_keys.append("box")
        all_keys = content_keys + meta_keys + special_keys
        return f"<Frame (Dataset) with keys: {all_keys}>"

    # Meta accessors
    @property
    def timestep(self) -> Optional[int]:
        return self._meta.get("timestep", None)

    @timestep.setter
    def timestep(self, value: Optional[int]):
        if value is not None and not isinstance(value, int):
            raise TypeError("Timestep must be an int or None")
        self._meta["timestep"] = value

    def get_meta(self, key: str, default=None):
        return self._meta.get(key, default)

    def set_meta(self, key: str, value: Any):
        self._meta[key] = value

    # Convenience methods
    def copy(self) -> "Frame":
        """Create a deep copy of the frame."""
        new_frame = self.__class__(box=Box.from_box(self.box) if self.box else None, meta=self._meta.copy())
        for key, value in self._data.items():
            new_frame._data[key] = value.copy(deep=True)
        return new_frame

    @classmethod
    def concat(cls, frames: Sequence["Frame"]) -> "Frame":
        """Concatenate multiple frames along the index dimension."""
        if not frames:
            return cls()
        new_box = Box.from_box(frames[0].box) if frames[0].box else None
        new_meta = frames[0]._meta.copy() if hasattr(frames[0], "_meta") else {}
        new_frame = cls(box=new_box, meta=new_meta)
        all_data_keys = set()
        for frame in frames:
            all_data_keys.update(frame._data.keys())
        for key in all_data_keys:
            datasets_to_concat = [frame._data[key] for frame in frames if key in frame._data]
            # 检查数据变量的dtype一致性
            if len(datasets_to_concat) > 1:
                first_ds = datasets_to_concat[0]
                for i, ds in enumerate(datasets_to_concat[1:], 1):
                    for var_name in first_ds.data_vars:
                        if var_name in ds.data_vars:
                            first_dtype = first_ds[var_name].dtype
                            current_dtype = ds[var_name].dtype
                            if first_dtype != current_dtype:
                                raise ValueError(f"Cannot concat datasets for key '{key}': "
                                               f"Variable '{var_name}' has dtype {first_dtype} in frame 0 "
                                               f"but dtype {current_dtype} in frame {i}")
            try:
                concatenated = xr.concat(datasets_to_concat, dim="index")
                new_frame._data[key] = concatenated
            except Exception as e:
                raise RuntimeError(f"Failed to concat datasets for key '{key}': {e}")
        return new_frame

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to a nested dictionary.
        
        Datasets are converted to dictionaries of their data variables.
        Box is converted if it has a to_dict method.
        """
        result_dict = {}
        
        # Convert Datasets to dictionaries of {var_name: var_values_as_numpy_array}
        for key, ds in self._data.items():
            result_dict[key] = {var: ds[var].values for var in ds.data_vars}
        
        # Add top-level scalar attributes
        result_dict.update(self._meta)
        
        # Add box if present and serializable
        if self.box is not None:
            if hasattr(self.box, 'to_dict') and callable(self.box.to_dict):
                result_dict["box"] = self.box.to_dict()
            else:
                # If no to_dict, store raw box or its repr. For simplicity, store as is.
                result_dict["box"] = self.box 
        # Add timestep if present
        if self.timestep is not None:
            result_dict["timestep"] = self.timestep
        return result_dict

    def __add__(self, other: "Frame") -> "Frame":
        """Concatenate this frame with another frame using the + operator."""
        if not isinstance(other, Frame):
            return NotImplemented # Or raise TypeError
        return self.concat([self, other])

    def __mul__(self, n: int) -> "Frame":
        """Replicate frame n times by concatenating n deep copies of itself."""
        if not isinstance(n, int) or n < 0:
            raise ValueError("Frame replication factor must be a non-negative integer.")
        if n == 0:
            return self.__class__() # Return an empty frame
        return self.concat([self.copy() for _ in range(n)])
