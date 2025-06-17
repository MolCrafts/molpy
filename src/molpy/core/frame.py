import numpy as np
import xarray as xr
from collections.abc import MutableMapping
from typing import Any, Sequence, Union, Dict, Optional

from .box import Box
from .forcefield import ForceField


def _dict_to_dataset(data: Dict[str, Any]) -> xr.Dataset:
    """Convert a mapping of arrays to an xarray.Dataset.
    
    All fields are treated equally as data variables. Dimensions are auto-generated
    based on array shapes without making assumptions about coordinate semantics.
    """
    if not data:
        # Return empty Dataset
        return xr.Dataset()
    
    data_vars = {}
    
    # First pass: determine the maximum first dimension size
    max_size = 0
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim >= 1:
            max_size = max(max_size, arr.shape[0])
    
    # If no arrays found, create a dataset with scalar values only
    if max_size == 0:
        for k, v in data.items():
            arr = np.asarray(v)
            # All scalars - create simple data variables without dimensions
            data_vars[k] = arr.item() if arr.ndim == 0 else arr
        return xr.Dataset(data_vars)
    
    # Second pass: create DataArrays with systematic dimension naming
    for k, v in data.items():
        arr = np.asarray(v)
        
        if arr.ndim == 0:
            # Scalar - store as scalar data variable
            data_vars[k] = arr.item()
        elif arr.ndim == 1:
            # 1D array - use generic dimension name
            data_vars[k] = (f"dim_{k}_0", arr)
        else:
            # Multi-dimensional arrays - create systematic dimension names
            dims = [f"dim_{k}_{i}" for i in range(arr.ndim)]
            data_vars[k] = (dims, arr)
    
    return xr.Dataset(data_vars)


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
        self._meta: Dict[str, Any] = {}
        
        # Use property setters for validation
        self.box = box
        self.forcefield = forcefield
        
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
        # 不再允许通过字典访问设置box和forcefield
        if key in ["box", "forcefield"]:
            raise ValueError(f"'{key}' should be set as an attribute (frame.{key} = value), not as a dictionary key")
        
        # meta keys (e.g. timestep, etc.)
        if key in self._meta or np.isscalar(value):
            self._meta[key] = value
            return
        if isinstance(value, xr.Dataset):
            self._data[key] = value
        elif isinstance(value, dict):
            self._data[key] = _dict_to_dataset(value)
        else:
            # Try to import pandas and handle DataFrame
            try:
                import pandas as pd
                if isinstance(value, pd.DataFrame):
                    # Convert DataFrame to dict and then to Dataset
                    df_dict = {}
                    for col in value.columns:
                        df_dict[col] = value[col].values
                    self._data[key] = _dict_to_dataset(df_dict)
                    return
            except ImportError:
                pass
            raise TypeError(f"Frame values for key '{key}' must be xarray.Dataset, dict (for _dict_to_dataset), pandas.DataFrame, or scalar. Got {type(value)}")

    def __delitem__(self, key: str) -> None:
        if key in ["box", "forcefield"]:
            raise ValueError(f"'{key}' should be deleted as an attribute (del frame.{key}), not as a dictionary key")
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
        if self.forcefield is not None:
            yield "forcefield"
        if self.timestep is not None:
            yield "timestep"

    def __len__(self) -> int:
        n = len(self._data) + len(self._meta)
        if self.box is not None:
            n += 1
        if self.forcefield is not None:
            n += 1
        return n

    def __repr__(self) -> str:
        content_keys = list(self._data.keys())
        meta_keys = list(self._meta.keys())
        special_keys = []
        if self.box is not None:
            special_keys.append("box")
        if self.forcefield is not None:
            special_keys.append("forcefield")
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

    # Box property with validation
    @property
    def box(self) -> Optional[Box]:
        """Get the simulation box."""
        return getattr(self, '_box', None)
    
    @box.setter
    def box(self, value: Optional[Box]) -> None:
        """Set the simulation box with validation."""
        if value is not None and not isinstance(value, Box):
            raise TypeError(f"Box must be a Box instance, got {type(value)}")
        self._box = value
    
    @box.deleter
    def box(self) -> None:
        """Delete the simulation box."""
        self._box = None
    
    # Forcefield property with validation
    @property
    def forcefield(self) -> Optional[ForceField]:
        """Get the force field."""
        return getattr(self, '_forcefield', None)
    
    @forcefield.setter
    def forcefield(self, value: Optional[ForceField]) -> None:
        """Set the force field with validation."""
        if value is not None and not isinstance(value, ForceField):
            raise TypeError(f"Forcefield must be a ForceField instance, got {type(value)}")
        self._forcefield = value
    
    @forcefield.deleter
    def forcefield(self) -> None:
        """Delete the force field."""
        self._forcefield = None

    # Convenience methods
    def copy(self) -> "Frame":
        """Create a deep copy of the frame."""
        # Create new frame with copied attributes
        box_copy = None
        if self.box is not None:
            if hasattr(self.box, 'copy'):
                box_copy = self.box.copy()
            elif hasattr(Box, 'from_box'):
                box_copy = Box.from_box(self.box)
            else:
                box_copy = self.box  # Fallback if no copy method
        
        forcefield_copy = None
        if self.forcefield is not None:
            if hasattr(self.forcefield, 'copy'):
                forcefield_copy = self.forcefield.copy()
            else:
                forcefield_copy = self.forcefield  # Assume immutable
        
        new_frame = self.__class__(
            box=box_copy,
            forcefield=forcefield_copy,
            meta=self._meta.copy()
        )
        for key, value in self._data.items():
            new_frame._data[key] = value.copy(deep=True)
        return new_frame

    @classmethod
    def concat(cls, frames: Sequence["Frame"]) -> "Frame":
        """Concatenate multiple frames along their first dimensions."""
        if not frames:
            return cls()
        new_box = Box.from_box(frames[0].box) if frames[0].box else None
        new_forcefield = frames[0].forcefield  # Take forcefield from first frame
        new_meta = frames[0]._meta.copy() if hasattr(frames[0], "_meta") else {}
        new_frame = cls(box=new_box, forcefield=new_forcefield, meta=new_meta)
        all_data_keys = set()
        for frame in frames:
            all_data_keys.update(frame._data.keys())
        
        for key in all_data_keys:
            datasets_to_concat = [frame._data[key] for frame in frames if key in frame._data]
            
            if len(datasets_to_concat) == 1:
                new_frame._data[key] = datasets_to_concat[0].copy(deep=True)
                continue
                
            # 检查数据变量的dtype一致性
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
                # 尝试沿着第一个共同维度拼接
                # 找到所有数据变量的第一个维度
                common_dims = None
                for ds in datasets_to_concat:
                    for var_name, var in ds.data_vars.items():
                        if var.dims:  # 如果有维度
                            first_dim = var.dims[0]
                            if common_dims is None:
                                common_dims = first_dim
                            elif common_dims != first_dim:
                                # 如果第一个维度不一致，使用第一个找到的维度
                                pass
                
                if common_dims:
                    concatenated = xr.concat(datasets_to_concat, dim=common_dims)
                else:
                    # 如果没有找到共同维度，直接使用第一个dataset
                    concatenated = datasets_to_concat[0].copy(deep=True)
                    
                new_frame._data[key] = concatenated
            except Exception as e:
                raise RuntimeError(f"Failed to concat datasets for key '{key}': {e}")
        return new_frame

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to a complete dictionary representation.
        
        This method leverages xarray's built-in to_dict() functionality for efficient
        and consistent serialization of datasets.
        
        Returns
        -------
        dict
            Complete dictionary representation of the Frame, including:
            - data: All datasets converted using xarray.Dataset.to_dict()
            - metadata: All frame metadata (excluding box and forcefield)
            - box: Simulation box (if present)
            - forcefield: Force field information (if present)
            - version: Format version for compatibility
        """
        result = {
            'version': '1.0',
            'data': {},
            'metadata': {}
        }
        
        # Convert datasets using xarray's built-in to_dict method
        for key, dataset in self._data.items():
            # xarray.Dataset.to_dict() handles all the serialization complexity
            result['data'][key] = dataset.to_dict()
        
        # Convert metadata (excluding box and forcefield which are handled separately)
        for key, value in self._meta.items():
            if isinstance(value, np.ndarray):
                result['metadata'][key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result['metadata'][key] = value.item()
            elif hasattr(value, 'to_dict') and callable(value.to_dict):
                result['metadata'][key] = value.to_dict()
            else:
                result['metadata'][key] = value
        
        # Store box as a separate top-level field
        if hasattr(self, 'box') and self.box is not None:
            # Use the {matrix, pbc, origin} format instead of Box.to_dict()
            result['box'] = {
                'matrix': self.box.matrix.tolist(),
                'pbc': self.box.pbc.tolist(),
                'origin': self.box.origin.tolist()
            }
                
        # Store forcefield as a separate top-level field
        if hasattr(self, 'forcefield') and self.forcefield is not None:
            if hasattr(self.forcefield, 'to_dict') and callable(self.forcefield.to_dict):
                result['forcefield'] = self.forcefield.to_dict()
            else:
                result['forcefield'] = str(self.forcefield)
        
        return result

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

    def to_hdf5_bytes(self, compression: str = 'gzip') -> bytes:
        """Convert Frame to HDF5 format as bytes buffer.
        
        This method creates an HDF5 representation in memory and returns it as
        bytes, which is perfect for network transmission or embedding in other formats.
        
        Parameters
        ----------
        compression : str, default 'gzip'
            Compression algorithm to use
            
        Returns
        -------
        bytes
            HDF5 data as bytes buffer
        """
        try:
            import h5py
            import io
        except ImportError:
            raise ImportError("h5py is required for HDF5 serialization. Install with: pip install h5py")
        
        # Get the core dictionary representation
        frame_dict = self.to_dict()
        
        # Create HDF5 file in memory
        buffer = io.BytesIO()
        
        with h5py.File(buffer, 'w') as f:
            # Store version and metadata
            f.attrs['molvis_frame_version'] = frame_dict['version']
            f.attrs['creation_time'] = str(np.datetime64('now'))
            
            # Store datasets
            data_group = f.create_group('data')
            for dataset_name, dataset_data in frame_dict['data'].items():
                ds_group = data_group.create_group(dataset_name)
                
                for var_name, values in dataset_data.items():
                    # Convert values to numpy array for HDF5 storage
                    arr = np.array(values)
                    
                    # Handle string arrays - convert to fixed-length strings for HDF5
                    if arr.dtype.kind == 'U':
                        # Find max string length and convert to fixed-length bytes
                        if arr.size > 0:
                            if arr.ndim == 0:
                                # Single string
                                str_val = str(arr.item())
                                arr = np.array(str_val.encode('utf-8'), dtype=f'S{len(str_val)}')
                            else:
                                # Array of strings
                                str_vals = [str(x) for x in arr.flat]
                                max_len = max(len(s) for s in str_vals) if str_vals else 1
                                # Convert to bytes array
                                byte_vals = [s.encode('utf-8') for s in str_vals]
                                arr = np.array(byte_vals, dtype=f'S{max_len}').reshape(arr.shape)
                    
                    # Create dataset with compression
                    ds_group.create_dataset(
                        var_name,
                        data=arr,
                        compression=compression,
                        compression_opts=9 if compression == 'gzip' else None
                    )
            
            # Store schema information
            schema_group = f.create_group('schema')
            for dataset_name, schema_data in frame_dict['schema'].items():
                ds_schema_group = schema_group.create_group(dataset_name)
                
                # Store variable schemas
                var_group = ds_schema_group.create_group('variables')
                for var_name, var_schema in schema_data['variables'].items():
                    var_subgroup = var_group.create_group(var_name)
                    var_subgroup.attrs['dtype'] = var_schema['dtype']
                    var_subgroup.attrs['shape'] = var_schema['shape']
                    var_subgroup.attrs['dimensions'] = [s.encode('utf-8') for s in var_schema['dimensions']]
                
                # Store coordinate schemas
                coord_group = ds_schema_group.create_group('coordinates')
                for coord_name, coord_schema in schema_data['coordinates'].items():
                    coord_subgroup = coord_group.create_group(coord_name)
                    coord_subgroup.attrs['dtype'] = coord_schema['dtype']
                    coord_subgroup.attrs['shape'] = coord_schema['shape']
            
            # Store coordinates
            coord_group = f.create_group('coordinates')
            for dataset_name, coord_data in frame_dict['coordinates'].items():
                ds_coord_group = coord_group.create_group(dataset_name)
                for coord_name, coord_values in coord_data.items():
                    ds_coord_group.create_dataset(coord_name, data=coord_values, compression=compression)
            
            # Store metadata
            meta_group = f.create_group('metadata')
            for key, value in frame_dict['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    meta_group.attrs[key] = value
                elif isinstance(value, list):
                    meta_group.create_dataset(key, data=value, compression=compression)
                elif isinstance(value, dict):
                    # For nested dicts, create subgroups
                    subgroup = meta_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (str, int, float, bool)):
                            subgroup.attrs[subkey] = subvalue
                        else:
                            subgroup.create_dataset(subkey, data=subvalue)
                else:
                    meta_group.attrs[key] = str(value)
            
            # Store box information as a separate group
            if 'box' in frame_dict:
                box_group = f.create_group('box')
                box_data = frame_dict['box']
                if isinstance(box_data, dict):
                    for key, value in box_data.items():
                        if isinstance(value, (list, np.ndarray)):
                            box_group.create_dataset(key, data=np.array(value), compression=compression)
                        else:
                            box_group.attrs[key] = value
                else:
                    box_group.attrs['string_representation'] = str(box_data)
        
        # Get the bytes data
        buffer.seek(0)
        return buffer.getvalue()

    @classmethod
    def from_hdf5_bytes(cls, data: bytes) -> 'Frame':
        """Create Frame from HDF5 bytes buffer.
        
        Parameters
        ----------
        data : bytes
            HDF5 data as bytes
            
        Returns
        -------
        Frame
            Reconstructed Frame object
        """
        try:
            import h5py
            import io
        except ImportError:
            raise ImportError("h5py is required for HDF5 serialization. Install with: pip install h5py")
        
        buffer = io.BytesIO(data)
        
        with h5py.File(buffer, 'r') as f:
            # Reconstruct the dictionary format
            frame_dict = {
                'version': f.attrs.get('molvis_frame_version', '1.0'),
                'data': {},
                'schema': {},
                'coordinates': {},
                'metadata': {}
            }
            
            # Load datasets
            if 'data' in f:
                data_group = f['data']
                for dataset_name in data_group.keys():
                    ds_group = data_group[dataset_name]
                    frame_dict['data'][dataset_name] = {}
                    
                    for var_name in ds_group.keys():
                        data_array = ds_group[var_name][:]
                        # Convert back to lists for consistency, handle bytes strings
                        if isinstance(data_array, np.ndarray):
                            if data_array.dtype.kind == 'S':  # Bytes strings
                                if data_array.ndim == 0:
                                    # Single bytes string
                                    frame_dict['data'][dataset_name][var_name] = data_array.item().decode('utf-8')
                                else:
                                    # Array of bytes strings
                                    frame_dict['data'][dataset_name][var_name] = [
                                        item.decode('utf-8') if isinstance(item, bytes) else str(item)
                                        for item in data_array.flat
                                    ]
                                    # Reshape if needed
                                    if data_array.shape != (len(frame_dict['data'][dataset_name][var_name]),):
                                        # For multi-dimensional string arrays, keep as 1D list for now
                                        pass
                            else:
                                frame_dict['data'][dataset_name][var_name] = data_array.tolist()
                        else:
                            frame_dict['data'][dataset_name][var_name] = data_array
            
            # Load schema
            if 'schema' in f:
                schema_group = f['schema']
                for dataset_name in schema_group.keys():
                    ds_schema_group = schema_group[dataset_name]
                    frame_dict['schema'][dataset_name] = {
                        'variables': {},
                        'coordinates': {}
                    }
                    
                    if 'variables' in ds_schema_group:
                        var_group = ds_schema_group['variables']
                        for var_name in var_group.keys():
                            var_subgroup = var_group[var_name]
                            frame_dict['schema'][dataset_name]['variables'][var_name] = {
                                'dtype': var_subgroup.attrs.get('dtype', ''),
                                'shape': list(var_subgroup.attrs.get('shape', [])),
                                'dimensions': [s.decode('utf-8') if isinstance(s, bytes) else s 
                                             for s in var_subgroup.attrs.get('dimensions', [])]
                            }
                    
                    if 'coordinates' in ds_schema_group:
                        coord_group = ds_schema_group['coordinates']
                        for coord_name in coord_group.keys():
                            coord_subgroup = coord_group[coord_name]
                            frame_dict['schema'][dataset_name]['coordinates'][coord_name] = {
                                'dtype': coord_subgroup.attrs.get('dtype', ''),
                                'shape': list(coord_subgroup.attrs.get('shape', []))
                            }
            
            # Load coordinates
            if 'coordinates' in f:
                coord_group = f['coordinates']
                for dataset_name in coord_group.keys():
                    ds_coord_group = coord_group[dataset_name]
                    frame_dict['coordinates'][dataset_name] = {}
                    
                    for coord_name in ds_coord_group.keys():
                        coord_data = ds_coord_group[coord_name][:]
                        frame_dict['coordinates'][dataset_name][coord_name] = coord_data.tolist()
            
            # Load metadata
            if 'metadata' in f:
                meta_group = f['metadata']
                
                # Load attributes
                for key, value in meta_group.attrs.items():
                    if isinstance(value, bytes):
                        frame_dict['metadata'][key] = value.decode('utf-8')
                    else:
                        frame_dict['metadata'][key] = value
                
                # Load datasets
                for key in meta_group.keys():
                    if isinstance(meta_group[key], h5py.Group):
                        # Handle nested groups
                        subdict = {}
                        subgroup = meta_group[key]
                        for subkey, subvalue in subgroup.attrs.items():
                            subdict[subkey] = subvalue
                        for subkey in subgroup.keys():
                            subdict[subkey] = subgroup[subkey][:]
                        frame_dict['metadata'][key] = subdict
                    else:
                        frame_dict['metadata'][key] = meta_group[key][:].tolist()
            
            # Load box information
            if 'box' in f:
                box_group = f['box']
                box_dict = {}
                
                # Load attributes
                for key, value in box_group.attrs.items():
                    if isinstance(value, bytes):
                        box_dict[key] = value.decode('utf-8')
                    else:
                        box_dict[key] = value
                
                # Load datasets
                for key in box_group.keys():
                    box_dict[key] = box_group[key][:].tolist()
                
                frame_dict['box'] = box_dict
        
        # Use from_dict to reconstruct the Frame
        return cls.from_dict(frame_dict)

    def save(self, filename: str, format: str = 'hdf5', **kwargs):
        """Save Frame to file in specified format.
        
        Parameters
        ----------
        filename : str
            Output filename
        format : str, default 'hdf5'
            File format ('hdf5', 'json', 'pickle')
        **kwargs
            Additional arguments passed to format-specific methods
        """
        if format.lower() in ['hdf5', 'h5']:
            # Use the bytes method and write to file
            hdf5_bytes = self.to_hdf5_bytes(**kwargs)
            with open(filename, 'wb') as f:
                f.write(hdf5_bytes)
        elif format.lower() == 'json':
            import json
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif format.lower() == 'pickle':
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'hdf5', 'json', or 'pickle'")

    @classmethod  
    def load(cls, filename: str, format: Optional[str] = None) -> 'Frame':
        """Load Frame from file.
        
        Parameters
        ----------
        filename : str
            Input filename
        format : str, optional
            File format. If None, inferred from filename extension
            
        Returns
        -------
        Frame
            Loaded Frame object
        """
        if format is None:
            # Infer format from extension
            if filename.endswith(('.hdf5', '.h5')):
                format = 'hdf5'
            elif filename.endswith('.json'):
                format = 'json'
            elif filename.endswith(('.pkl', '.pickle')):
                format = 'pickle'
            else:
                raise ValueError(f"Cannot infer format from filename: {filename}")
        
        if format.lower() in ['hdf5', 'h5']:
            # Read file as bytes and use bytes method
            with open(filename, 'rb') as f:
                hdf5_bytes = f.read()
            return cls.from_hdf5_bytes(hdf5_bytes)
        elif format.lower() == 'json':
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        elif format.lower() == 'pickle':
            import pickle
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Frame':
        """Create Frame from dictionary representation.
        
        This method leverages xarray's from_dict() functionality for efficient
        deserialization of datasets.
        
        Parameters
        ----------
        data : dict
            Dictionary representation from to_dict()
            
        Returns
        -------
        Frame
            Reconstructed Frame object
        """
        # Check version compatibility
        version = data.get('version', '1.0')
        if version != '1.0':
            raise ValueError(f"Unsupported Frame format version: {version}")
        
        # Extract box and forcefield from top-level fields (new format)
        box_data = data.get('box', None)
        forcefield_data = data.get('forcefield', None)
        
        # Create frame with reconstructed attributes
        frame = cls()
        
        # Reconstruct box if present
        if box_data is not None:
            if isinstance(box_data, dict):
                # Handle box dict format: {matrix: (3, 3), pbc: (3, ), origin: (3, )}
                try:
                    matrix = np.array(box_data['matrix'])
                    pbc = np.array(box_data['pbc'])
                    origin = np.array(box_data['origin'])
                    frame.box = Box(matrix=matrix, pbc=pbc, origin=origin)
                except Exception:
                    # Fallback to storing as metadata
                    frame._meta['box_data'] = box_data
            else:
                # Store as metadata if not a dict
                frame._meta['box_data'] = box_data
        
        # Reconstruct forcefield if present
        if forcefield_data is not None:
            if isinstance(forcefield_data, dict):
                # Try to reconstruct ForceField from dict
                try:
                    if hasattr(ForceField, 'from_dict'):
                        frame.forcefield = ForceField.from_dict(forcefield_data)
                    else:
                        # Store as metadata for manual reconstruction
                        frame._meta['forcefield_data'] = forcefield_data
                except Exception:
                    # Fallback to storing as metadata
                    frame._meta['forcefield_data'] = forcefield_data
            else:
                # Store as metadata if not a dict
                frame._meta['forcefield_data'] = forcefield_data
        
        # Reconstruct datasets using xarray's from_dict method
        frame_data = data.get('data', {})
        for key, dataset_dict in frame_data.items():
            # Use xarray.Dataset.from_dict() to reconstruct the dataset
            frame._data[key] = xr.Dataset.from_dict(dataset_dict)
        
        # Restore metadata (excluding box and forcefield which are now handled separately)
        metadata = data.get('metadata', {})
        for key, value in metadata.items():
            # Skip box and forcefield in metadata (legacy format compatibility)
            if key in ['box', 'forcefield']:
                # Check if we haven't already processed these from top-level
                if box_data is None and key == 'box':
                    frame._meta['box_data'] = value
                elif forcefield_data is None and key == 'forcefield':
                    frame._meta['forcefield_data'] = value
            else:
                frame._meta[key] = value
        
        return frame
