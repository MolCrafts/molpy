import csv
from collections import defaultdict
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Any, TypeVar

from .frame import Block

# --- TypeBucket generic implementation ---

T = TypeVar("T")


def get_nearest_type[T](item: T) -> type[T]:
    """Get the concrete type of an object"""
    return type(item)


class TypeBucket[T]:
    """
    Generic TypeBucket implementation that groups and stores objects by their concrete type.

    Supports two storage modes:
    - Set mode (set): For storing unique type definitions with automatic deduplication
    - List mode (list): For storing entity objects while maintaining order

    Usage example:
        # Set mode (used by forcefield)
        bucket = TypeBucket(container_type=set)

        # List mode (used by entity)
        bucket = TypeBucket(container_type=list)
    """

    def __init__(self, container_type: type = set) -> None:
        """
        Initialize TypeBucket

        Args:
            container_type: Container type, set (deduplication) or list (maintain order)
        """
        self.container_type = container_type
        if container_type == set:
            self._items: dict[type[T], Any] = defaultdict(set)
        else:
            self._items: dict[type[T], Any] = defaultdict(list)

    def add(self, item: T) -> None:
        """Add an object to its corresponding type bucket"""
        cls = get_nearest_type(item)
        if self.container_type == set:
            self._items[cls].add(item)
        else:
            self._items[cls].append(item)

    def add_many(self, items: Any) -> None:
        """Add multiple objects"""
        for item in items:
            self.add(item)

    def remove(self, item: T) -> bool:
        """
        Remove an object from the bucket

        Returns:
            True if successfully removed, False otherwise
        """
        cls = get_nearest_type(item)
        bucket = self._items.get(cls)
        if not bucket:
            return False

        if self.container_type == set:
            if item in bucket:
                bucket.discard(item)
                if not bucket:
                    self._items.pop(cls, None)
                return True
            return False
        else:
            # List mode: use identity comparison (is) not equality comparison (==)
            for i, obj in enumerate(bucket):
                if obj is item:
                    bucket.pop(i)
                    if not bucket:
                        self._items.pop(cls, None)
                    return True
            return False

    def bucket(self, cls: type[T]) -> list[T]:
        """
        Get objects of the specified type and all its subclasses

        Args:
            cls: Target type

        Returns:
            List containing all matching objects
        """
        result: list[T] = []
        for k, items in self._items.items():
            if isinstance(k, type) and issubclass(k, cls):
                result.extend(items)
        return result

    def exact_bucket(self, cls: type[T]) -> list[T]:
        """
        Get objects of exact type (excluding subclasses)

        Args:
            cls: Target type

        Returns:
            List containing exact matching objects
        """
        bucket = self._items.get(cls)
        return list(bucket) if bucket else []

    def classes(self) -> Iterator[type[T]]:
        """Return all concrete types currently stored"""
        return iter(self._items.keys())

    def all(self) -> list[T]:
        """Return all objects from all buckets"""
        result: list[T] = []
        for bucket in self._items.values():
            result.extend(bucket)
        return result

    def __len__(self) -> int:
        """Return total count of objects in all buckets"""
        return sum(len(b) for b in self._items.values())

    def __getitem__(self, cls: type[T]) -> list[T]:
        """Get exact type bucket (excluding subclasses)"""
        return self.exact_bucket(cls)

    def __setitem__(self, cls: type[T], items: Any) -> None:
        """Set bucket for specified type"""
        if self.container_type == set:
            self._items[cls] = set(items)
        else:
            self._items[cls] = list(items)


def to_dict_of_list(list_of_dict: list[dict[str, Any]]) -> dict[str, list[Any]]:
    result = defaultdict(list)
    for item in list_of_dict:
        for key, value in item.items():
            result[key].append(value)
    return dict(result)


def to_list_of_dict(list_of_dict: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(list_of_dict.keys())
    length = len(next(iter(list_of_dict.values())))
    return [{key: list_of_dict[key][i] for key in keys} for i in range(length)]


def read_csv(file: Path | StringIO, delimiter: str = ",") -> Block:
    """
    Read a CSV file or StringIO object and return a Block object.

    Args:
        file: Path to the CSV file or a StringIO object containing CSV data.
        delimiter: Delimiter used in the CSV file (default is comma).

    Returns:
        Block: A Block object containing the data from the CSV.
    """
    if isinstance(file, StringIO):
        file.seek(0)  # Ensure we read from the start
        reader = csv.DictReader(file, delimiter=delimiter)
        data = [row for row in reader]
    else:
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist.")

        with open(file, newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            data = [row for row in reader]

    # Convert list of dicts to dict of lists for Block
    dict_of_lists = to_dict_of_list(data)

    # Handle empty data case - create empty arrays for each column
    if not dict_of_lists and isinstance(file, StringIO):
        file.seek(0)
        reader = csv.DictReader(file, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        dict_of_lists = {field: [] for field in fieldnames}
    elif not dict_of_lists and not isinstance(file, StringIO):
        with open(file, newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            dict_of_lists = {field: [] for field in fieldnames}

    return Block(dict_of_lists)
