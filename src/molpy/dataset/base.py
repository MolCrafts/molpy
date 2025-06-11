
from typing import cast, Generic, Iterable, Optional, TypeVar, Union  # noqa: UP035


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_dict = dict[str, _T_co]
_T_tuple = tuple[_T_co, ...]
_T_stack = TypeVar("_T_stack", _T_tuple, _T_dict)


class Dataset(Generic[_T_co]):

    def __getitem__(self, index) -> _T_co:
        """
        Get an item from the dataset by index.
        """
        raise NotImplementedError("Subclasses must implement __getitem__ method.")
    
class RemoteDataset(Dataset[_T_co]):
    """
    A dataset that is remote and can be accessed via a URL.
    """
    def __init__(self, name: str, base_url: str, version: str|None = None, version_dev="master", env=None, registry=None, urls=None, retry_if_failed=0, allow_updates=True):
        import pooch
        self._goodboy = pooch.create(
            path=pooch.os_cache(name),
            base_url=base_url,
            version_dev=version_dev,
            env=env,
            registry=registry,
            urls=urls,
            retry_if_failed=retry_if_failed,
            allow_updates=allow_updates,
        )