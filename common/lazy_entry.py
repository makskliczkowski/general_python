"""Lazy-loaded entries for HDF5/NPZ/Pickle/JSON files."""

from __future__ import annotations

from    pathlib import Path
from    typing import Any, Dict, Iterable
import  json
import  pickle
import  numpy as np

# ------------------------------------------------------

class LazyDataEntry:
    """
    Base class for lazy data entries.
    """

    __slots__ = ("filepath", "filename", "params", "_cache", "_known_keys")

    def __init__(self, filepath: str, params: Dict[str, Any]):
        ''' Initialization of a lazy data entry. '''
        self.filepath   = filepath
        self.filename   = Path(filepath).name
        self.params     = params
        # Cache for loaded data items, keyed by dataset name or "default" for single-item entries.
        self._cache     : Dict[str, Any] = {}
        self._known_keys: tuple[str, ...] | None = None

    def __getitem__(self, key: str):
        ''' Access a dataset by key, loading it if not already cached. '''
        if key in self._cache:
            return self._cache[key]
        self._load_item(key)
        if key not in self._cache:
            raise KeyError(f"Key '{key}' not found in {self.filename}")
        return self._cache[key]

    def _load_item(self, key: str):
        self.load_all()

    def __contains__(self, key: str):
        if key in self._cache:
            return True
        if self._known_keys is None:
            known = self._list_keys()
            if known is not None:
                self._known_keys = tuple(known)
        if self._known_keys is not None:
            return key in self._known_keys
        try:
            self._load_item(key)
            return key in self._cache
        except KeyError:
            return False

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(filename='{self.filename}', n_cached={len(self._cache)})"

    # Convenience methods for dict-like access and introspection

    def get(self, key: str, default=None):
        ''' Get a dataset by key, returning default if not found. '''
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def keys(self):
        ''' Return available dataset keys, using known keys if available, otherwise loading all data. '''
        if self._cache:
            return self._cache.keys()
        if self._known_keys is None:
            known = self._list_keys()
            if known is not None:
                self._known_keys = tuple(known)
        if self._known_keys is not None:
            return self._known_keys
        self.load_all()
        return self._cache.keys()

    def values(self):
        ''' Return loaded dataset values, loading all data if not already cached. '''
        if not self._cache:
            self.load_all()
        return self._cache.values()

    def items(self):
        ''' Return loaded dataset items, loading all data if not already cached. '''
        if not self._cache:
            self.load_all()
        return self._cache.items()

    def load(self, keys: str | Iterable[str] | None = None):
        """Load one key, multiple keys, or all keys into cache."""
        if keys is None:
            return self.load_all()
        if isinstance(keys, str):
            _ = self[keys]
            return self
        for key in keys:
            _ = self[key]
        return self

    def is_loaded(self, key: str | None = None) -> bool:
        ''' Check if a specific key or any key is loaded in the cache. '''
        if key is None:
            return bool(self._cache)
        return key in self._cache

    def clear_cache(self, keys: str | Iterable[str] | None = None):
        ''' Clear cached data for one key, multiple keys, or all keys. '''
        if keys is None:
            self._cache.clear()
            return self
        if isinstance(keys, str):
            self._cache.pop(keys, None)
            return self
        for key in keys:
            self._cache.pop(key, None)
        return self

    def require(self, key: str):
        """Strict key accessor, explicit in client code."""
        return self[key]

    def get_many(self, keys: Iterable[str], default=None) -> Dict[str, Any]:
        ''' Get multiple datasets by keys, returning a dict of key-value pairs, using default for missing keys. '''
        return {k: self.get(k, default) for k in keys}

    def to_dict(self, keys: Iterable[str] | None = None, copy: bool = False) -> Dict[str, Any]:
        ''' Return a dict of datasets for specified keys, loading them if necessary. If keys is None, return all datasets. If copy is True, return a new dict; otherwise return the internal cache dict (which may be shared). '''
        if keys is None:
            self.load_all()
            out = self._cache
        else:
            out = {k: self[k] for k in keys}
        if not copy:
            return out
        return dict(out)

    # Convenience methods for treating single-dataset entries as arrays

    def _default_array_key(self) -> str:
        ''' Determine the default key to use when treating this entry as an array. If there is a single known key, use that. If there is a "default" key, use that. Otherwise, raise an error to avoid ambiguity. '''
        keys = list(self.keys())
        if "default" in keys:
            return "default"
        if len(keys) == 1:
            return keys[0]
        raise ValueError(f"Cannot infer array key for {self.filename}. Available keys: {keys}. Pass key explicitly.")

    # Convenience methods for array-like access when the entry is known to contain a single dataset or has a "default" key.

    def as_array(self, key: str | None = None):
        key = self._default_array_key() if key is None else key
        return np.asarray(self[key])

    def __array__(self, dtype=None):
        arr = self.as_array()
        if dtype is None:
            return arr
        return np.asarray(arr, dtype=dtype)

    # Optional methods for entries that support introspection without loading full data

    def shape(self, key: str | None = None):
        return self.as_array(key=key).shape

    def dtype(self, key: str | None = None):
        return self.as_array(key=key).dtype

    def load_all(self):
        raise NotImplementedError

    def _list_keys(self):
        """Optional fast path for listing keys without loading full payload."""
        return None

# ------------------------------------------------------

class LazyHDF5Entry(LazyDataEntry):
    """Lazy loader for HDF5 datasets."""
    __slots__ = ()

    def _load_item(self, key: str):
        from .hdf5man import HDF5Manager

        data = HDF5Manager.read_hdf5(self.filepath, keys=[key], verbose=False)
        if key in data:
            self._cache[key] = data[key]
            if self._known_keys is not None and key not in self._known_keys:
                self._known_keys = tuple((*self._known_keys, key))
            return
        raise KeyError(f"Key '{key}' not found in {self.filename}")

    def load_all(self):
        from .hdf5man import HDF5Manager

        self._cache = HDF5Manager.read_hdf5(self.filepath, verbose=False)
        self._cache.pop("filename", None)
        self._known_keys = tuple(self._cache.keys())
        return self

    def _list_keys(self):
        if self._known_keys is not None:
            return self._known_keys
        try:
            import h5py
            dataset_keys = []
            with h5py.File(self.filepath, "r") as hf:
                hf.visititems(
                    lambda _name, obj: dataset_keys.append(obj.name)
                    if isinstance(obj, h5py.Dataset)
                    else None
                )
            self._known_keys = tuple(dataset_keys)
            return self._known_keys
        except Exception:
            return None

    def shape(self, key: str | None = None):
        if key is None:
            key = self._default_array_key()
        if key in self._cache:
            return np.asarray(self._cache[key]).shape
        try:
            import h5py
            with h5py.File(self.filepath, "r") as hf:
                return tuple(hf[key].shape)
        except KeyError as exc:
            raise KeyError(f"Key '{key}' not found in {self.filename}") from exc

    def dtype(self, key: str | None = None):
        if key is None:
            key = self._default_array_key()
        if key in self._cache:
            return np.asarray(self._cache[key]).dtype
        try:
            import h5py
            with h5py.File(self.filepath, "r") as hf:
                return np.dtype(hf[key].dtype)
        except KeyError as exc:
            raise KeyError(f"Key '{key}' not found in {self.filename}") from exc

# ------------------------------------------------------

class LazyNpzEntry(LazyDataEntry):
    """Lazy loader for .npz files."""
    __slots__ = ()

    def _load_item(self, key: str):
        if self._known_keys is not None and key not in self._known_keys:
            raise KeyError(f"Key '{key}' not found in {self.filename}")
        try:
            with np.load(self.filepath) as data:
                if key in data:
                    self._cache[key] = data[key]
                    return
        except Exception as exc:
            raise KeyError(f"Error loading {key} from {self.filename}: {exc}") from exc
        raise KeyError(f"Key '{key}' not found in {self.filename}")

    def load_all(self):
        with np.load(self.filepath) as data:
            for key in data.files:
                self._cache[key] = data[key]
        self._known_keys = tuple(self._cache.keys())
        return self

    def _list_keys(self):
        if self._known_keys is not None:
            return self._known_keys
        try:
            with np.load(self.filepath) as data:
                self._known_keys = tuple(data.files)
                return self._known_keys
        except Exception:
            return None

# ------------------------------------------------------

class LazyPickleEntry(LazyDataEntry):
    """Lazy loader for .pkl/.pickle files."""
    __slots__ = ()

    def load_all(self):
        try:
            with open(self.filepath, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self._cache.update(data)
                else:
                    self._cache["default"] = data
                self._known_keys = tuple(self._cache.keys())
        except Exception:
            pass
        return self

# ------------------------------------------------------

class LazyJsonEntry(LazyDataEntry):
    """Lazy loader for .json files."""
    __slots__ = ()

    def load_all(self):
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for key, value in data.items():
                        try:
                            self._cache[key] = np.array(value)
                        except Exception:
                            self._cache[key] = value
                else:
                    try:
                        self._cache["default"] = np.array(data)
                    except Exception:
                        self._cache["default"] = data
                self._known_keys = tuple(self._cache.keys())
        except Exception:
            pass
        return self

# ------------------------------------------------------

__all__ = [
    "LazyDataEntry",
    "LazyHDF5Entry",
    "LazyNpzEntry",
    "LazyPickleEntry",
    "LazyJsonEntry",
]

# ------------------------------------------------------
#! EOF
# ------------------------------------------------------
