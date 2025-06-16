import numpy as np
import xarray as xr
from collections.abc import MutableMapping
from typing import Any, Sequence, Union, Dict, Optional

from .box import Box
from .forcefield import ForceField


def _dict_to_dataarray(data: Dict[str, np.ndarray]) -> xr.DataArray:
    """Convert a mapping of arrays to an ``xarray.DataArray``.
    
    Non-scalar arrays must have the same length (first dimension).
    Scalar arrays become scalar coordinates.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping variable names to numpy arrays or scalars.
        
    Returns
    -------
    xr.DataArray
        DataArray with the arrays/scalars as coordinates.
        
    Raises
    ------
    ValueError
        If non-scalar arrays have different first dimension lengths.
    """
    if not data:
        return xr.DataArray([], dims=["index"], name="empty_data")

    # Convert data to numpy arrays, handling ragged arrays
    arrays = {}
    for k, v in data.items():
        try:
            arrays[k] = np.asarray(v)
        except ValueError as e:
            if "inhomogeneous" in str(e) or "sequence" in str(e):
                # Handle ragged arrays (like atom_indices lists)
                arrays[k] = np.array(v, dtype=object)
            else:
                raise

    non_scalar_arrays = {k: v for k, v in arrays.items() if v.ndim > 0}
    scalar_arrays = {k: v for k, v in arrays.items() if v.ndim == 0}

    coords = {}
    length = -1

    if non_scalar_arrays:
        first_arr_val = next(iter(non_scalar_arrays.values()))
        length = first_arr_val.shape[0]

        for name, arr in non_scalar_arrays.items():
            if arr.shape[0] != length:
                raise ValueError(
                    f"All non-scalar arrays must have the same first dimension length. "
                    f"Array '{name}' has length {arr.shape[0]}, expected {length}."
                )
            if arr.ndim == 1:
                coords[name] = ("index", arr)
            elif arr.ndim == 2:
                # Handle 2D arrays like position (N, 3)
                if name == "position" or arr.shape[1] == 3:
                    coords[name] = (("index", "spatial"), arr)
                else:
                    # For other 2D arrays, flatten the extra dimensions
                    coords[name] = (("index", f"{name}_dim"), arr)
            else:
                # For higher dimensional arrays, flatten them
                reshaped = arr.reshape(length, -1)
                coords[name] = (("index", f"{name}_dim"), reshaped)
    
    for name, scalar_arr in scalar_arrays.items():
        coords[name] = scalar_arr.item() # Store as scalar coordinate

    if length == -1: # All scalars or empty non_scalar_arrays
        if scalar_arrays and not non_scalar_arrays: # Only scalars provided
            return xr.DataArray(None, coords=coords, name="scalar_data")
        else:
            length = 0

    # Create data array with indices as data if length > 0
    main_data = np.arange(length) if length > 0 else np.array([])
    
    # Determine dimensions for the main DataArray by finding all unique dimensions from coordinates
    all_dims = set()
    for coord_name, coord_data in coords.items():
        if isinstance(coord_data, tuple):
            coord_dims, _ = coord_data
            if isinstance(coord_dims, tuple):
                all_dims.update(coord_dims)
            else:
                all_dims.add(coord_dims)
    
    # Ensure 'index' is always first
    dims = ["index"]
    for dim in sorted(all_dims):
        if dim != "index":
            dims.append(dim)
    
    # Create main data with appropriate shape based on dimensions
    if length > 0 and len(dims) > 1:
        # For multi-dimensional case, we need to create data with the right shape
        shape = [length]
        for dim in dims[1:]:
            if dim == "spatial":
                shape.append(3)  # Standard spatial dimension size
            else:
                # Find the size from coordinates
                for coord_name, coord_data in coords.items():
                    if isinstance(coord_data, tuple):
                        coord_dims, coord_arr = coord_data
                        if isinstance(coord_dims, tuple) and dim in coord_dims:
                            dim_idx = coord_dims.index(dim)
                            shape.append(coord_arr.shape[dim_idx])
                            break
                else:
                    shape.append(1)  # Default size if not found
        main_data = np.zeros(shape)
    
    return xr.DataArray(
        main_data, 
        dims=dims,
        coords=coords,
        name="constructed_data"
    )


class Frame(MutableMapping):
    """Container of simulation data based on :class:`xarray.DataArray`."""
    def __init__(self, data: Optional[Dict[str, Union[Dict[str, np.ndarray], xr.DataArray]]] = None, 
                 *, box: Optional[Box] = None, forcefield: Optional[ForceField] = None, meta: Optional[Dict[str, Any]] = None, **extra_meta):
        """Initialize Frame.
        
        Parameters
        ----------
        data : dict, optional
            Dictionary mapping keys to either:
            - Dict[str, np.ndarray]: Dictionary of arrays/scalars. Non-scalar arrays must have the same first dimension length.
            - xr.DataArray: Pre-built DataArray
        box : Box, optional
            Simulation box
        """
        self._data: Dict[str, xr.DataArray] = {}
        self.box = box
        self._meta: Dict[str, Any] = {}
        if meta is not None:
            self._meta.update(meta)
        self._meta.update(extra_meta)
        # Remove legacy self.timestep
        # self.timestep: Optional[int] = None
        
        if data:
            for key, value in data.items():
                self[key] = value # Calls __setitem__

    # Mapping protocol
    def __getitem__(self, key: str) -> xr.DataArray:
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
        if isinstance(value, xr.DataArray):
            self._data[key] = value
        elif isinstance(value, dict):
            self._data[key] = _dict_to_dataarray(value)
        else:
            raise TypeError(
                f"Frame values for key '{key}' must be xarray.DataArray, dict (for _dict_to_dataarray), "
                f"or scalar. Got {type(value)}"
            )

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
        return f"<Frame with keys: {all_keys}>"

    # Meta accessors
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
            arrays_to_concat = []
            ref_array_structure = None
            for frame in frames:
                if key in frame._data:
                    ref_array_structure = frame._data[key]
                    break
            if ref_array_structure is None and all_data_keys:
                pass
            for frame in frames:
                if key in frame._data:
                    arrays_to_concat.append(frame._data[key])
                elif ref_array_structure is not None:
                    empty_coords = {}
                    for coord_name, coord_da in ref_array_structure.coords.items():
                        if coord_name == ref_array_structure.dims[0]:
                            empty_coords[coord_name] = (ref_array_structure.dims[0], np.array([], dtype=coord_da.dtype))
                        else:
                            if coord_da.ndim == 0:
                                empty_coords[coord_name] = coord_da.copy()
                            else:
                                empty_coord_shape = list(coord_da.shape)
                                if ref_array_structure.dims[0] in coord_da.dims:
                                    idx_dim_pos = coord_da.dims.index(ref_array_structure.dims[0])
                                    empty_coord_shape[idx_dim_pos] = 0
                                    empty_coords[coord_name] = (coord_da.dims, np.empty(empty_coord_shape, dtype=coord_da.dtype))
                                else:
                                    empty_coords[coord_name] = coord_da.copy()
                    empty_main_data_shape = list(ref_array_structure.shape)
                    empty_main_data_shape[0] = 0
                    empty_da = xr.DataArray(
                        np.empty(empty_main_data_shape, dtype=ref_array_structure.dtype),
                        dims=ref_array_structure.dims,
                        coords=empty_coords,
                        name=ref_array_structure.name
                    )
                    arrays_to_concat.append(empty_da)
            if arrays_to_concat:
                try:
                    concatenated_array = xr.concat(arrays_to_concat, dim="index")
                    new_frame._data[key] = concatenated_array
                except Exception as e:
                    print(f"Warning: Could not concatenate DataArrays for key '{key}'. Error: {e}")
                    print(f"Details: Attempted to concat {len(arrays_to_concat)} arrays.")
        return new_frame

    def split(self, key: str, column: str) -> list["Frame"]:
        """Split frame into multiple frames based on unique values in a coordinate of a DataArray.
        
        Parameters
        ----------
        key : str
            The data key (e.g., 'atoms') corresponding to a DataArray in the frame.
        column : str
            The name of the coordinate within self[key] DataArray to use for splitting.
            
        Returns
        -------
        list[Frame]
            List of new frames, each corresponding to a unique value in the specified column.
        """
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in frame data. Available keys: {list(self._data.keys())}")
        
        data_array_to_split = self._data[key]
        if column not in data_array_to_split.coords:
            raise KeyError(f"Coordinate '{column}' not found in DataArray '{key}'. Available coordinates: {list(data_array_to_split.coords.keys())}")
        
        split_coord_values = data_array_to_split.coords[column].values
        unique_split_values = np.unique(split_coord_values)
        
        resulting_frames = []
        for unique_val in unique_split_values:
            # Create a boolean mask for the current unique value along the 'index' dimension
            # Ensure the mask is 1D and aligned with 'index'
            if split_coord_values.ndim == 1:
                 mask = data_array_to_split.coords[column] == unique_val
            else: # If coord is multi-dim but tied to index, this needs care. Assume 1D coord for splitting.
                 raise ValueError(f"Splitting on multi-dimensional coordinate '{column}' is not directly supported. Ensure it's a 1D coordinate.")

            # Create a new frame for this segment
            new_segment_frame = self.__class__(box=Box.from_box(self.box) if self.box else None, meta=self._meta.copy())

            # Filter each DataArray in self._data based on the mask
            for da_key, da_value in self._data.items():
                if "index" in da_value.dims and len(da_value["index"]) > 0 : # Check if da_value is indexable by 'index' and not empty
                    try:
                        # Ensure mask is compatible with this DataArray's index
                        # If splitting coord was from a different DataArray, this might fail.
                        # For simplicity, assume mask is universally applicable or split logic is on a "primary" DataArray.
                        # The mask is derived from data_array_to_split, so it has the same 'index'.
                        new_segment_frame._data[da_key] = da_value.isel(index=mask)
                    except Exception as e:
                        print(f"Warning: Could not split DataArray '{da_key}' using mask from '{key}.{column}'. Error: {e}")
                        new_segment_frame._data[da_key] = da_value.copy(deep=True) # or skip, or add empty
                else: # DataArray not indexed by "index" or is empty, copy as is
                    new_segment_frame._data[da_key] = da_value.copy(deep=True)
            
            resulting_frames.append(new_segment_frame)
        
        return resulting_frames

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to a nested dictionary.
        
        DataArrays are converted to dictionaries of their coordinates.
        Box is converted if it has a to_dict method.
        """
        result_dict = {}
        
        # Convert DataArrays to dictionaries of {coord_name: coord_values_as_numpy_array}
        for key, data_array_val in self._data.items():
            da_dict = {}
            for coord_name, coord_data in data_array_val.coords.items():
                da_dict[coord_name] = coord_data.values # .values gives numpy array
            result_dict[key] = da_dict
        
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

    @property
    def timestep(self) -> Optional[int]:
        return self._meta.get("timestep", None)

    @timestep.setter
    def timestep(self, value: Optional[int]):
        if value is not None and not isinstance(value, int):
            raise TypeError("Timestep must be an int or None")
        self._meta["timestep"] = value