# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2024-02-05
# version: 0.0.1

from typing import Optional, List, Dict, Any, Iterator, Union
from copy import deepcopy
from .frame import Frame


class Trajectory:
    """
    A trajectory represents a time-ordered sequence of molecular configurations (frames).
    
    This class stores loaded frames in memory and provides an interface for lazy loading
    and caching of trajectory data. It is designed to work with TrajectoryReader for
    on-demand frame loading.
    
    Each frame contains atomic positions, velocities, forces, and system properties 
    at a specific time step.
    """
    
    def __init__(self, frames: Optional[List[Frame]] = None, max_cache_size: Optional[int] = None, **meta):
        """
        Initialize a trajectory.
        
        Args:
            frames: Optional list of Frame objects. If None, creates an empty trajectory.
            max_cache_size: Maximum number of frames to keep in memory. If None, no limit.
            **meta: Additional metadata for the trajectory.
        """
        self.frames: Dict[int, Frame] = {}  # frame_index -> Frame
        self._frame_indices: List[int] = []  # ordered list of available frame indices
        self._total_frames: Optional[int] = None  # total number of frames (when known)
        self._max_cache_size = max_cache_size
        self._access_order: List[int] = []  # for LRU cache management
        self.meta = meta
        
        if frames:
            for i, frame in enumerate(frames):
                self._add_frame(i, frame)
    
    def __len__(self) -> int:
        """Return the number of available frames in the trajectory."""
        if self._total_frames is not None:
            return self._total_frames
        return len(self._frame_indices)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Frame, 'Trajectory']:
        """
        Get a frame by index or slice.
        
        Args:
            idx: Frame index or slice
            
        Returns:
            Frame object at the specified index or new Trajectory with sliced frames
            
        Raises:
            IndexError: If frame is not available and cannot be loaded
            KeyError: If frame index is not in the trajectory
        """
        if isinstance(idx, slice):
            # Handle slice - return new Trajectory with loaded frames in the slice range
            start, stop, step = idx.indices(len(self))
            sliced_frames = []
            for i in range(start, stop, step):
                if i in self.frames:
                    sliced_frames.append(self.frames[i])
                else:
                    raise KeyError(f"Frame {i} not loaded. Use TrajectoryReader.load_frame() first.")
            return Trajectory(sliced_frames, max_cache_size=self._max_cache_size, **self.meta)
        
        # Handle single index
        if idx < 0:
            # Handle negative indexing
            if self._total_frames is not None:
                idx = self._total_frames + idx
            else:
                idx = len(self._frame_indices) + idx
                
        if idx in self.frames:
            # Update access order for LRU
            if idx in self._access_order:
                self._access_order.remove(idx)
            self._access_order.append(idx)
            return self.frames[idx]
        
        # Frame not loaded - caller should use TrajectoryReader to load it
        raise KeyError(f"Frame {idx} not loaded. Use TrajectoryReader.load_frame() first.")
    
    def append(self, frame: Frame) -> None:
        """
        Add a frame to the trajectory.
        
        Args:
            frame: Frame object to add
        """
        next_index = max(self._frame_indices, default=-1) + 1
        self._add_frame(next_index, frame)
    
    def extend(self, frames: List[Frame]) -> None:
        """
        Add multiple frames to the trajectory.
        
        Args:
            frames: List of Frame objects to add
        """
        for frame in frames:
            self.append(frame)
    
    def _add_frame(self, index: int, frame: Frame) -> None:
        """
        Internal method to add a frame at a specific index.
        
        Args:
            index: Frame index
            frame: Frame object to add
        """
        # If frame already exists, just update access order
        if index in self.frames:
            if index in self._access_order:
                self._access_order.remove(index)
            self._access_order.append(index)
            return
            
        # Manage cache size before adding new frame
        if self._max_cache_size and len(self.frames) >= self._max_cache_size:
            self._evict_lru_frame()
        
        self.frames[index] = frame
        if index not in self._frame_indices:
            self._frame_indices.append(index)
            self._frame_indices.sort()
        
        # Update access order
        self._access_order.append(index)
    
    def _evict_lru_frame(self) -> None:
        """Remove the least recently used frame from cache."""
        if not self._access_order:
            return
        
        lru_index = self._access_order.pop(0)
        if lru_index in self.frames:
            del self.frames[lru_index]
            # Note: We don't remove from _frame_indices as it tracks 
            # which frames are available, not which are loaded
    
    def need_more(self, total_frames: int) -> bool:
        """
        Check if more frames need to be loaded.
        
        Args:
            total_frames: Total number of frames available
            
        Returns:
            True if more frames are available to load
        """
        self._total_frames = total_frames
        return len(self._frame_indices) < total_frames
    
    def is_loaded(self, index: int) -> bool:
        """
        Check if a frame is already loaded in memory.
        
        Args:
            index: Frame index to check
            
        Returns:
            True if frame is loaded, False otherwise
        """
        return index in self.frames
    
    def get_loaded_indices(self) -> List[int]:
        """
        Get list of frame indices that are currently loaded.
        
        Returns:
            Sorted list of loaded frame indices
        """
        return sorted(self.frames.keys())
    
    def clear_cache(self) -> None:
        """Remove all frames from memory cache."""
        self.frames.clear()
        self._access_order.clear()
        self._frame_indices.clear()
    
    def set_total_frames(self, total: int) -> None:
        """
        Set the total number of frames in the trajectory.
        
        Args:
            total: Total number of frames
        """
        self._total_frames = total
    
    def iterframes(self) -> Iterator[Frame]:
        """
        Iterate over loaded frames in the trajectory.
        
        Note: This only iterates over frames currently in memory.
        Use TrajectoryReader for complete iteration.
        """
        for index in sorted(self.frames.keys()):
            yield self.frames[index]
    
    def copy(self) -> "Trajectory":
        """Create a copy of the trajectory with all loaded frames."""
        new_traj = Trajectory(max_cache_size=self._max_cache_size, **deepcopy(self.meta))
        new_traj._total_frames = self._total_frames
        for index, frame in self.frames.items():
            new_traj._add_frame(index, deepcopy(frame))
        return new_traj
    
    def __iter__(self):
        """
        Iterate over loaded frames in the trajectory.
        
        Note: This only iterates over frames currently in memory.
        Use TrajectoryReader for complete iteration.
        """
        return self.iterframes()
    
    def __repr__(self) -> str:
        loaded = len(self.frames)
        total = self._total_frames or len(self._frame_indices)
        return f"<Trajectory {loaded}/{total} frames loaded, meta={self.meta}>"
    
    # Placeholder for IO (legacy compatibility)
    def to_file(self, filename, fmt=None):
        raise NotImplementedError("Trajectory export not implemented. Use TrajectoryWriter instead.")

    @classmethod
    def from_file(cls, filename, fmt=None):
        raise NotImplementedError("Trajectory import not implemented. Use TrajectoryReader instead.")

    @classmethod
    def concat(cls, trajectories: list["Trajectory"]) -> "Trajectory":
        """Concatenate multiple Trajectory objects into one."""
        if not trajectories:
            return cls()
        
        # Create new trajectory with combined metadata
        meta = trajectories[0].meta.copy() if hasattr(trajectories[0], "meta") else {}
        new_traj = cls(**meta)
        
        # Add all frames from all trajectories
        for traj in trajectories:
            for frame in traj.iterframes():
                new_traj.append(frame)
        
        return new_traj

    def to_dict(self) -> dict:
        """Convert the trajectory to a dict with frame data and meta info."""
        return {
            "frames": [frame.to_dict() for frame in self.iterframes()],
            "meta": self.meta.copy(),
            "loaded_indices": self.get_loaded_indices(),
            "total_frames": self._total_frames
        }

    def get_meta(self, key: str, default=None):
        """Get metadata value."""
        return self.meta.get(key, default)

    def set_meta(self, key: str, value: Any):
        """Set metadata value."""
        self.meta[key] = value