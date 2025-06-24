import numpy as np
import xarray as xr
from collections.abc import MutableMapping
from typing import Any, Sequence, Union, Dict, Optional

from .box import Box
from .forcefield import ForceField


def _dict_to_dataset(data: Dict[str, Any], component_name: str = "atoms") -> xr.Dataset:
    """Convert a mapping of arrays to an xarray.Dataset.
    
    All fields are treated equally as data variables. Uses consistent dimension
    naming to avoid alignment issues during concatenation.
    
    Args:
        data: Dictionary of arrays/scalars
        component_name: Name of the component (e.g., 'atoms', 'bonds') for dimension naming
    """
    if not data:
        # Return empty Dataset
        return xr.Dataset()
    
    data_vars = {}
    
    # First pass: determine the maximum first dimension size (number of entities)
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
    
    # Use consistent dimension naming for all components
    # The first dimension is always the entity index (e.g., atom index)
    primary_dim = f"{component_name}_id"
    
    # Second pass: create DataArrays with consistent dimension naming
    for k, v in data.items():
        arr = np.asarray(v)
        
        if arr.ndim == 0:
            # Scalar - store as scalar data variable
            data_vars[k] = arr.item()
        elif arr.ndim == 1:
            # 1D array - use primary dimension
            if arr.shape[0] == max_size:
                data_vars[k] = (primary_dim, arr)
            else:
                # Different size - create specific dimension
                data_vars[k] = (f"{k}_dim", arr)
        else:
            # Multi-dimensional arrays - first dim is primary, others are specific
            dims = [primary_dim] + [f"{k}_dim_{i}" for i in range(1, arr.ndim)]
            if arr.shape[0] == max_size:
                data_vars[k] = (dims, arr)
            else:
                # Different first dimension size - use specific naming
                dims = [f"{k}_dim_{i}" for i in range(arr.ndim)]
                data_vars[k] = (dims, arr)
    
    return xr.Dataset(data_vars)


class Frame(MutableMapping):
    """Container of simulation data based on :class:`xarray.DataTree`."""
    def __init__(self, data: Optional[Dict[str, Union[Dict[str, Any], xr.Dataset]]] = None, 
                 *, box: Optional[Box] = None, forcefield: Optional[ForceField] = None, meta: Optional[Dict[str, Any]] = None, **extra_meta):
        """Initialize Frame.
        
        Parameters
        ----------
        data : dict, optional
            Dictionary mapping keys to either:
            - Dict[str, np.ndarray]: Dictionary of arrays/scalars. Non-scalar arrays must have the same first dimension length.
            - xr.Dataset: Pre-built Dataset
            - Dict with xarray format: Dictionary from Dataset.to_dict() with keys 'data_vars', 'coords', 'dims', 'attrs'
        box : Box, optional
            Simulation box
        """
        # Use DataTree for hierarchical data organization
        self._data = xr.DataTree(name="root")
        
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
    def __getitem__(self, key: str):
        """Get item from Frame.
        
        Returns:
        - For DataTree children: Returns the underlying Dataset (mutable)
        - For meta keys: Returns the meta value directly
        """
        if key in self._data.children:
            # Return the actual Dataset, not DatasetView, so it's mutable
            return self._data[key].to_dataset()
        elif key in self._meta:
            return self._meta[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in Frame with simplified logic around DataTree core."""
        if key in ["box", "forcefield"]:
            raise ValueError(f"'{key}' should be set as an attribute (frame.{key} = value), not as a dictionary key")
        
        # Scalar values directly to meta
        if self._is_scalar_value(value):
            self._meta[key] = value
            return
        
        # Existing meta keys continue to meta
        if key in self._meta:
            self._meta[key] = value
            return
        
        # Convert all data to Dataset and store in DataTree
        dataset = self._to_dataset(value, key)
        self._data[key] = xr.DataTree(dataset, name=key)

    def _is_scalar_value(self, value: Any) -> bool:
        """Check if value should be stored as scalar metadata."""
        return (np.isscalar(value) or 
                isinstance(value, (str, int, float, bool, list)) or
                (isinstance(value, dict) and not self._is_data_dict(value)))

    def _is_data_dict(self, value: dict) -> bool:
        """Check if dictionary contains array data."""
        if all(k in value for k in ['data_vars', 'coords', 'dims', 'attrs']):
            return True  # xarray format
        
        return any(isinstance(v, (list, np.ndarray)) and len(np.asarray(v).shape) >= 1 
                   for v in value.values())

    def _to_dataset(self, value: Any, key: str) -> xr.Dataset:
        """Unified conversion to xarray.Dataset."""
        if isinstance(value, xr.Dataset):
            return value
        
        elif isinstance(value, dict):
            if self._is_xarray_dict_format(value):
                return xr.Dataset.from_dict(value)
            else:
                return _dict_to_dataset(value, component_name=key)
        
        else:
            # Handle pandas DataFrame and other types
            try:
                import pandas as pd
                if isinstance(value, pd.DataFrame):
                    df_dict = {col: value[col].values for col in value.columns}
                    return _dict_to_dataset(df_dict, component_name=key)
            except ImportError:
                pass
            
            raise TypeError(f"Cannot convert {type(value)} to Dataset")

    def _is_xarray_dict_format(self, value: dict) -> bool:
        """Check if dictionary is in xarray.Dataset.to_dict() format."""
        return all(k in value for k in ['data_vars', 'coords', 'dims', 'attrs'])

    def __delitem__(self, key: str) -> None:
        if key in ["box", "forcefield"]:
            raise ValueError(f"'{key}' should be deleted as an attribute (del frame.{key}), not as a dictionary key")
        elif key in self._meta:
            del self._meta[key]
        elif key in self._data.children:
            del self._data[key]
        else:
            raise KeyError(key)

    def __iter__(self):
        yield from self._data.children.keys()
        yield from self._meta.keys()
        if self.box is not None:
            yield "box"
        if self.forcefield is not None:
            yield "forcefield"
        if self.timestep is not None:
            yield "timestep"

    def __len__(self) -> int:
        n = len(self._data.children) + len(self._meta)
        if self.box is not None:
            n += 1
        if self.forcefield is not None:
            n += 1
        return n

    def __repr__(self) -> str:
        content_keys = list(self._data.children.keys())
        meta_keys = list(self._meta.keys())
        special_keys = []
        if self.box is not None:
            special_keys.append("box")
        if self.forcefield is not None:
            special_keys.append("forcefield")
        all_keys = content_keys + meta_keys + special_keys
        return f"<Frame (DataTree) with keys: {all_keys}>"

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
            # Box typically stores numpy arrays that can be copied
            import copy
            box_copy = copy.deepcopy(self.box)
        
        forcefield_copy = None
        if self.forcefield is not None:
            # ForceField is typically immutable or has complex structure
            import copy
            forcefield_copy = copy.deepcopy(self.forcefield)
        
        new_frame = self.__class__(
            box=box_copy,
            forcefield=forcefield_copy,
            meta=self._meta.copy()
        )
        for key in self._data.children.keys():
            new_frame._data[key] = xr.DataTree(self._data[key].ds.copy(deep=True), name=key)
        return new_frame

    @classmethod
    def concat(cls, frames: Sequence["Frame"]) -> "Frame":
        """Concatenate multiple frames along their first dimensions.
        
        Based on DataTree core, letting xarray handle complex alignment logic.
        """
        if not frames:
            return cls()
            
        # Inherit attributes from first frame
        new_frame = cls(
            box=frames[0].box,
            forcefield=frames[0].forcefield,
            meta=frames[0]._meta.copy()
        )
        
        # Collect all data keys from all frames
        all_keys = set()
        for frame in frames:
            all_keys.update(frame._data.children.keys())
        
        # Concatenate each key using xarray's concat
        for key in all_keys:
            datasets = [frame._data[key].ds for frame in frames 
                       if key in frame._data.children]
            
            if datasets:
                # Use xarray's concat, let it handle dimension alignment
                primary_dim = f"{key}_id"
                try:
                    concatenated = xr.concat(datasets, dim=primary_dim)
                    new_frame._data[key] = xr.DataTree(concatenated, name=key)
                except Exception as e:
                    # If dimension mismatch, try reindexing
                    aligned_datasets = []
                    total_entities = 0
                    
                    for ds in datasets:
                        # Determine entity count from dataset
                        entity_count = ds.sizes.get(primary_dim, 0)
                        if entity_count == 0:
                            # Infer entity count from largest dimension
                            entity_count = max(ds.sizes.values()) if ds.sizes else 0
                        
                        # Create new coordinates for this dataset
                        new_coords = np.arange(total_entities, total_entities + entity_count)
                        aligned_ds = ds.assign_coords({primary_dim: new_coords})
                        aligned_datasets.append(aligned_ds)
                        total_entities += entity_count
                    
                    concatenated = xr.concat(aligned_datasets, dim=primary_dim)
                    new_frame._data[key] = xr.DataTree(concatenated, name=key)
        
        return new_frame

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert frame to a complete dictionary representation using unified interface.
        
        This method leverages xarray's built-in to_dict() functionality and
        the unified to_dict interface of all molpy components for consistent 
        serialization.
        
        Returns
        -------
        dict
            Complete dictionary representation of the Frame, including:
            - __class__: Class information for reconstruction
            - data: All datasets converted using xarray.Dataset.to_dict()
            - metadata: All frame metadata recursively serialized
            - box: Simulation box using Box.to_dict() (if present)
            - forcefield: Force field using ForceField.to_dict() (if present)
            - version: Format version for compatibility
        """
        result = {
            '__class__': f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            'version': '1.0',
            'data': {},
            'metadata': {}
        }
        
        # Convert datasets using xarray's built-in to_dict method
        for key in self._data.children.keys():
            # xarray.Dataset.to_dict() handles all the serialization complexity
            result['data'][key] = self._data[key].ds.to_dict()
        
        # Convert metadata recursively, calling to_dict on nested components
        for key, value in self._meta.items():
            result['metadata'][key] = self._serialize_value(value)
        
        # Store box using unified interface
        if hasattr(self, 'box') and self.box is not None:
            result['box'] = self._serialize_value(self.box)
                
        # Store forcefield using unified interface
        if hasattr(self, 'forcefield') and self.forcefield is not None:
            result['forcefield'] = self._serialize_value(self.forcefield)
        
        return result

    def _serialize_value(self, value: Any) -> Any:
        """
        Recursively serialize values using unified to_dict interface.
        
        Automatically calls to_dict on objects that support it, otherwise
        handles numpy arrays and other common types.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value ready for JSON/dict storage
        """
        if hasattr(value, 'to_dict') and callable(value.to_dict):
            # Use the unified to_dict interface
            return value.to_dict()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        else:
            return value

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
        """
        Create Frame from dictionary using unified from_dict interface.
        
        This method leverages xarray's from_dict() functionality and the unified
        from_dict interface of all molpy components for consistent deserialization.
        
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
        
        # Create new frame instance
        frame = cls()
        
        # Reconstruct datasets using xarray's from_dict method
        frame_data = data.get('data', {})
        for key, dataset_dict in frame_data.items():
            # Use xarray.Dataset.from_dict() to reconstruct the dataset
            dataset = xr.Dataset.from_dict(dataset_dict)
            frame._data[key] = xr.DataTree(dataset, name=key)
        
        # Recursively reconstruct metadata using unified interface
        metadata = data.get('metadata', {})
        for key, value in metadata.items():
            frame._meta[key] = cls._deserialize_value(value)
        
        # Reconstruct box using unified interface
        box_data = data.get('box')
        if box_data is not None:
            frame.box = cls._deserialize_value(box_data, target_type=Box)
        
        # Reconstruct forcefield using unified interface  
        ff_data = data.get('forcefield')
        if ff_data is not None:
            frame.forcefield = cls._deserialize_value(ff_data, target_type=ForceField)
        
        return frame

    @classmethod
    def _deserialize_value(cls, value: Any, target_type: type = None) -> Any:
        """
        Recursively deserialize values using unified from_dict interface.
        
        Args:
            value: Value to deserialize
            target_type: Expected type (if known) that supports from_dict
            
        Returns:
            Deserialized value
        """
        if isinstance(value, dict):
            # If target type specified and has from_dict method, try it first
            if target_type and hasattr(target_type, 'from_dict') and callable(target_type.from_dict):
                try:
                    return target_type.from_dict(value)
                except Exception:
                    # If target type reconstruction fails, continue with generic approach
                    pass
            
            # Try to infer type from __class__ key and use appropriate from_dict
            if '__class__' in value:
                class_path = value['__class__']
                try:
                    # Import and get the class
                    module_name, class_name = class_path.rsplit(".", 1)
                    module = __import__(module_name, fromlist=[class_name])
                    target_class = getattr(module, class_name)
                    
                    # Call from_dict if available
                    if hasattr(target_class, 'from_dict') and callable(target_class.from_dict):
                        return target_class.from_dict(value)
                except (ImportError, AttributeError, ValueError):
                    # If class reconstruction fails, fall through to generic handling
                    pass
            
            # Recursively process regular dictionaries
            return {k: cls._deserialize_value(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [cls._deserialize_value(v) for v in value]
        
        else:
            # Return primitive values as-is
            return value

    def wrapped(self, key: str = "atoms", coord_field: str = "xyz") -> "Frame":
        """
        返回一个新Frame，其指定key下的坐标（如atoms/xyz）经过box.wrap处理。
        默认处理["atoms"]["xyz"]。
        """
        if self.box is None:
            raise ValueError("Frame.box is not set, cannot wrap coordinates.")
        self[key][coord_field] = (self[key][coord_field].dims, self.box.wrap(self[key][coord_field].values))
        # 返回新Frame，保留原有box和meta
        return self
