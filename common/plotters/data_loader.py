"""
Convenient data loading and filtering for plot-ready result entries.

Quick use:
    from QES.general_python.common.plotters.data_loader import load_results, filter_results

    results = load_results("/path/to/data", filters={"Lx": 4, "Ly": 4})
    results = filter_results(results, {"hx": ("between", (0.0, 1.0))})

    first = results[0]
    spectral = first["/spectral/akw"]  # lazy-loaded dataset

`load_results` supports HDF5/NPZ/Pickle/JSON files and returns lazy entries.
`filter_results` works with both object-style entries (`.params`) and dict-style entries.
`load_results` returns ``ResultSet`` (a list subclass) with convenient methods
such as ``filtered(...)``, ``show(...)``, and ``show_filtered(...)``.
"""

from __future__ import annotations

from    pathlib import Path
from    typing import Any, Callable, Iterable, List, Sequence, Union, TYPE_CHECKING
import  re

import numpy as np

try:
    from ..lazy_entry import (
        LazyDataEntry,
        LazyHDF5Entry,
        LazyJsonEntry,
        LazyNpzEntry,
        LazyPickleEntry,
    )
except ImportError:
    raise ImportError("Required lazy entry classes are missing. Ensure 'lazy_entry.py' is present in the same package.")

if TYPE_CHECKING:
    from ..flog import Logger

SUPPORTED_EXTENSIONS = {
    ".h5"       : "hdf5",
    ".hdf5"     : "hdf5",
    ".npz"      : "npz",
    ".pkl"      : "pickle",
    ".pickle"   : "pickle",
    ".json"     : "json",
}

# Union of all entry types returned by load_results and used in plotting utilities.
_ENTRY_TYPES        = Union[LazyDataEntry, "ResultProxy"]
_SUPPORTED_SUFFIXES = set(SUPPORTED_EXTENSIONS.keys())
_FILENAME_PARAM_RE  = re.compile(r"(\w+)=([\d\.\-]+)")
_SIZE_PARAM_RE      = re.compile(r"(Ns|Lx|Ly|Lz)=(\d+)")
_OTHER_PARAM_RE     = re.compile(r"(\w+)=([\d\.\-]+)")
_ENTRY_BY_TYPE      = {
    "hdf5"  : LazyHDF5Entry,
    "npz"   : LazyNpzEntry,
    "pickle": LazyPickleEntry,
    "json"  : LazyJsonEntry,
}

# ----------------------------------------
# Helper functions for loading and filtering results from files or in-memory data.
# ----------------------------------------

def _log(logger: "Logger" | None, level: str, message: str, color: str | None = None):
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if log_fn is None:
        return
    try:
        if color is None:
            log_fn(message)
        else:
            log_fn(message, color=color)
    except TypeError:
        log_fn(message)

def _to_number(value: Any):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))

def parse_filename(filename: str) -> dict:
    """Parse simple key=value parameters from filename."""
    params = {}
    for match in _FILENAME_PARAM_RE.finditer(filename):
        key, value = match.group(1), match.group(2)
        num = _to_number(value)
        params[key] = num if num is not None else value
    return params

def _extract_params_from_path(filepath: Path) -> dict:
    params          = parse_filename(filepath.name)

    for part in filepath.parts:
        for match in _SIZE_PARAM_RE.finditer(part):
            params.setdefault(match.group(1), int(match.group(2)))
        for match in _OTHER_PARAM_RE.finditer(part):
            key = match.group(1)
            num = _to_number(match.group(2))
            params.setdefault(key, num if num is not None else match.group(2))
    return params

# ----------------------------------------
# Main functions for loading and filtering results, and preparing data for plotting.
# ----------------------------------------

def _entry_from_path(filepath: Path, params: dict) -> LazyDataEntry | None:
    ftype = SUPPORTED_EXTENSIONS.get(filepath.suffix.lower())
    factory = _ENTRY_BY_TYPE.get(ftype)
    if factory is None:
        return None
    return factory(str(filepath), params)

def _iter_supported_files(path: Path, recursive: bool = True):
    if path.is_file():
        if path.suffix.lower() in _SUPPORTED_SUFFIXES:
            yield path
        return

    iterator = path.rglob("*") if recursive else path.glob("*")
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_SUFFIXES:
            yield candidate

def _params_for_result(result: Any, get_params_fun: Callable | None = None) -> dict:
    if get_params_fun is not None:
        return get_params_fun(result)
    if hasattr(result, "params"):
        return result.params
    if isinstance(result, dict):
        return result.get("params", result)
    return {}

def _eq_with_tol(lhs: Any, rhs: Any, tol: float) -> bool:
    if _is_numeric(lhs) and _is_numeric(rhs):
        return abs(float(lhs) - float(rhs)) <= tol
    return lhs == rhs

def _condition_match(param_value: Any, condition: Any, params: dict, tol: float) -> bool:
    '''
    Evaluate if a parameter value matches a given condition, which can be:
    - A scalar value (with numeric tolerance)
    - A list/tuple/set of values (any match with tolerance)
    - A tuple operator: ('eq'|'neq'|'lt'|'le'|'gt'|'ge', value)
    - A range operator: ('between', (min, max))
    - Membership operators: ('in'|'not_in', [v1, v2, ...])
    - String contains: ('contains', 'substring')
    - Callable: `lambda param_value, params: ...`
    - If condition is a callable, it is called with (param_value, params) and should return a boolean.
    
    Parameters
    ----------
    param_value:
        The value of the parameter to check.
    condition:
        The condition to check against, which can be various types as described above.
    params:
        The full parameter dictionary for the result, which can be used in callable conditions.
    tol:
        Numeric tolerance for equality checks.

    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise.

    '''
    
    
    if callable(condition):
        return bool(condition(param_value, params))

    if isinstance(condition, tuple) and len(condition) == 2 and isinstance(condition[0], str):
        op, raw_value = condition

        if isinstance(raw_value, str) and raw_value in params:
            value = params[raw_value]
        else:
            value = raw_value

        # Between : ('between', (min, max))
        if op == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("'between' requires (min, max).")
            vmin, vmax = value
            if _is_numeric(param_value) and _is_numeric(vmin) and _is_numeric(vmax):
                return float(vmin) - tol <= float(param_value) <= float(vmax) + tol
            return vmin <= param_value <= vmax

        # Membership : ('in', [v1, v2, ...]) or ('not_in', [v1, v2, ...])
        if op in {"in", "not_in"}:
            if not isinstance(value, (list, tuple, set, np.ndarray)):
                raise ValueError(f"'{op}' requires a list/tuple/set/array.")
            matches = any(_eq_with_tol(param_value, v, tol) for v in value)
            return matches if op == "in" else not matches

        # String contains : ('contains', 'substring')
        if op == "contains":
            return str(value) in str(param_value)

        # Comparison operators : ('eq', value), ('neq', value), ('lt', value), ('le', value), ('gt', value), ('ge', value)
        if op == "eq":
            return _eq_with_tol(param_value, value, tol)
        if op == "neq":
            return not _eq_with_tol(param_value, value, tol)

        # Numeric comparisons
        if op in {"lt", "le", "gt", "ge"}:
            if not (_is_numeric(param_value) and _is_numeric(value)):
                return False
            pv = float(param_value)
            vv = float(value)
            if op == "lt":
                return pv < vv
            if op == "le":
                return pv <= vv + tol
            if op == "gt":
                return pv > vv
            return pv >= vv - tol

        raise ValueError(f"Unknown filter operator: {op}")

    if isinstance(condition, (list, tuple, set, np.ndarray)):
        return any(_eq_with_tol(param_value, c, tol) for c in condition)

    return _eq_with_tol(param_value, condition, tol)

# ----------------------------------------
#! CONVENIENT RESULT CONTAINER
# ----------------------------------------

class ResultSet(list):
    """
    List-like container for result entries with convenience query/preview methods.

    Notes
    -----
    - Inherits from ``list`` to preserve normal list behavior.
    - Slice operations return ``ResultSet`` (not plain list).
    """

    __slots__ = ("get_params_fun", "tol", "name")

    def __init__(
        self,
        iterable        : Iterable[Any] = (),
        *,
        get_params_fun  : Callable | None = None,
        tol             : float = 1e-9,
        name            : str = "results",
    ):
        super().__init__(iterable)
        self.get_params_fun = get_params_fun
        self.tol            = float(tol)
        self.name           = name

    def __getitem__(self, item):
        ''' Override to return ResultSet for slices, preserving get_params_fun and tol. '''
        out = super().__getitem__(item)
        if isinstance(item, slice):
            return ResultSet(out, get_params_fun=self.get_params_fun, tol=self.tol, name=self.name,)
        return out

    def copy(self) -> "ResultSet":
        return ResultSet(self, get_params_fun=self.get_params_fun, tol=self.tol, name=self.name)

    def filtered(self, filters: dict | Callable[[Any], bool] | None = None, *, get_params_fun: Callable | None = None, tol: float | None = None) -> "ResultSet":
        ''' Filter results based on provided conditions. '''
        use_tol     = self.tol if tol is None else float(tol)
        extractor   = self.get_params_fun if get_params_fun is None else get_params_fun
        return filter_results(self, filters=filters, get_params_fun=extractor, tol=use_tol)

    def where(self, filters: dict | Callable[[Any], bool] | None = None, *, get_params_fun: Callable | None = None, tol: float | None = None) -> "ResultSet":
        ''' Alias for filtered() to allow chaining like results.where(...).show(...) '''
        return self.filtered(filters=filters, get_params_fun=get_params_fun, tol=tol)

    def param_values(self, key: str, *, default: Any = np.nan, get_params_fun: Callable | None = None,) -> np.ndarray:
        ''' Extract an array of parameter values for a given key across all results. '''
        extractor   = self.get_params_fun if get_params_fun is None else get_params_fun
        values      = []
        for item in self:
            params = _params_for_result(item, get_params_fun=extractor)
            values.append(params.get(key, default) if isinstance(params, dict) else default)
        return np.asarray(values)

    def unique(self, key: str, *, drop_nan: bool = True, get_params_fun: Callable | None = None) -> np.ndarray:
        ''' Get unique values of a parameter key across all results, with option to drop NaNs. '''
        
        values = self.param_values(key, get_params_fun=get_params_fun)
        if not drop_nan:
            return np.unique(values)
        try:
            arr = values.astype(float)
            return np.unique(arr[~np.isnan(arr)])
        except (TypeError, ValueError):
            return np.unique(values)

    def sort_by(self, key: str, *, reverse: bool = False, get_params_fun: Callable | None = None) -> "ResultSet":
        ''' Return a new ResultSet sorted by a parameter key. '''
        extractor = self.get_params_fun if get_params_fun is None else get_params_fun

        def _sort_key(item: Any):
            params = _params_for_result(item, get_params_fun=extractor)
            if not isinstance(params, dict):
                return (2, "")
            value = params.get(key, np.nan)
            if _is_numeric(value):
                return (0, float(value))
            if value is None:
                return (2, "")
            return (1, str(value))

        ordered = sorted(self, key=_sort_key, reverse=reverse)
        return ResultSet(ordered, get_params_fun=extractor, tol=self.tol, name=self.name)

    def first(self, default: Any = None):
        ''' Return the first entry or default if empty. '''
        return self[0] if self else default

    @staticmethod
    def _format_scalar(value: Any) -> str:
        ''' Format a scalar value for display, with special handling for numeric types. '''
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _default_fields(self, *, max_fields: int = 8) -> list[str]:
        ''' Heuristic to determine which parameter keys to show by default, based on frequency and a preferred order. '''
        preferred                   = ["Lx", "Ly", "Lz", "Ns", "J", "hx", "hy", "hz", "delta", "gamma", "beta", "T"]
        key_counts: dict[str, int]  = {}
        for item in self[: min(len(self), 64)]:
            params = _params_for_result(item, get_params_fun=self.get_params_fun)
            if not isinstance(params, dict):
                continue
            for k in params.keys():
                key_counts[k] = key_counts.get(k, 0) + 1
        if not key_counts:
            return []

        ordered     = [k for k in preferred if k in key_counts]
        remaining   = [k for k in key_counts.keys() if k not in ordered]
        remaining.sort(key=lambda k: (-key_counts[k], k))
        ordered.extend(remaining)
        return ordered[:max_fields]

    def show(
        self,
        *,
        fields: Sequence[str] | None = None,
        limit: int = 20,
        sort_by: str | None = None,
        reverse: bool = False,
        include_filename: bool = True,
    ) -> "ResultSet":
        ''' Display a tabular preview of the results in the console. '''
        
        if not self:
            print(f"{self.name}: <empty>")
            return self

        data    = self.sort_by(sort_by, reverse=reverse) if sort_by is not None else self
        n_total = len(data)
        n_show  = max(0, n_total if limit is None else min(int(limit), n_total))
        sample  = data[:n_show]

        fields_list = list(fields) if fields is not None else data._default_fields()
        headers     = ["#", "file"] + fields_list if include_filename else ["#"] + fields_list
        rows: list[list[str]] = []
        for idx, entry in enumerate(sample):
            row = [str(idx)]
            if include_filename:
                row.append(str(getattr(entry, "filename", getattr(entry, "filepath", "<entry>"))))
            params = _params_for_result(entry, get_params_fun=data.get_params_fun)
            for key in fields_list:
                value = params.get(key, "") if isinstance(params, dict) else ""
                row.append(self._format_scalar(value))
            rows.append(row)

        widths = [len(h) for h in headers]
        for row in rows:
            for i, text in enumerate(row):
                widths[i] = min(max(widths[i], len(text)), 48)

        def _clip(text: str, w: int) -> str:
            if len(text) <= w:
                return text
            if w <= 1:
                return text[:w]
            return text[: w - 1] + "…"

        def _line(cols: Sequence[str]) -> str:
            return " | ".join(_clip(c, widths[i]).ljust(widths[i]) for i, c in enumerate(cols))

        print(f"{self.name}: showing {n_show}/{n_total}")
        print(_line(headers))
        print("-+-".join("-" * w for w in widths))
        for row in rows:
            print(_line(row))
        return self

    def show_filtered(
        self,
        filters: dict | Callable[[Any], bool] | None = None,
        *,
        get_params_fun: Callable | None = None,
        tol: float | None = None,
        fields: Sequence[str] | None = None,
        limit: int = 20,
        sort_by: str | None = None,
        reverse: bool = False,
        include_filename: bool = True,
    ) -> "ResultSet":
        subset = self.filtered(filters=filters, get_params_fun=get_params_fun, tol=tol)
        subset.show(
            fields=fields,
            limit=limit,
            sort_by=sort_by,
            reverse=reverse,
            include_filename=include_filename,
        )
        return subset

# ----------------------------------------
#! Main functions for loading and filtering results, and preparing data for plotting.
# ----------------------------------------

def _as_result_set(results: Iterable[Any], *, get_params_fun: Callable | None = None, tol: float = 1e-9, name: str = "results") -> ResultSet:
    if isinstance(results, ResultSet):
        return ResultSet(
            results,
            get_params_fun=results.get_params_fun if get_params_fun is None else get_params_fun,
            tol=results.tol,
            name=results.name if results.name else name,
        )
    return ResultSet(results, get_params_fun=get_params_fun, tol=tol, name=name)

def filter_results(
    results         : Iterable[Any],
    filters         : dict | Callable[[Any], bool] | None = None,
    get_params_fun  : Callable | None = None,
    *,
    tol             : float = 1e-9,
) -> ResultSet:
    """
    Filter result entries by parameter conditions.

    Parameters
    ----------
    results:
        Iterable of result-like objects (with `.params` or dict-like).
    filters:
        - None: return all
        - callable: `filters(result) -> bool`
        - dict: `{param: condition}` where condition supports:
          - scalar exact match (numeric with tolerance)
          - list/tuple/set membership
          - tuple operators: ('eq'|'neq'|'lt'|'le'|'gt'|'ge', value)
          - range operator: ('between', (min, max))
          - membership operators: ('in'|'not_in', [v1, ...])
          - string contains: ('contains', 'sub')
          - callable: `lambda param_value, params: ...`
    get_params_fun:
        Optional extractor `f(result) -> dict`.
    tol:
        Numeric tolerance for equality-like checks.
    """
    source      = _as_result_set(results, get_params_fun=get_params_fun, tol=tol)
    extractor   = source.get_params_fun if get_params_fun is None else get_params_fun
    if filters is None:
        return source.copy()

    if callable(filters):
        return ResultSet(
            (r for r in source if filters(r)),
            get_params_fun=extractor,
            tol=tol,
            name=source.name,
        )

    filter_items        = tuple(filters.items())
    filtered            = []
    params_for_result   = _params_for_result
    condition_match     = _condition_match
    for result in source:
        params = params_for_result(result, get_params_fun=extractor)
        if not isinstance(params, dict):
            continue

        matches = True
        for key, condition in filter_items:
            if key not in params:
                matches = False
                break
            if not condition_match(params[key], condition, params, tol):
                matches = False
                break

        if matches:
            filtered.append(result)

    return ResultSet(filtered, get_params_fun=extractor, tol=tol, name=source.name)

def load_results(
    data_dir: str,
    *,
    filters: dict | Callable[[Any], bool] | None = None,
    lx=None,
    ly=None,
    lz=None,
    Ns=None,
    # Optional post-processing function to modify params after extraction, e.g. to compute derived parameters or convert units.
    post_process_func: Callable[[dict], None] | None = None,
    # Optional function to extract params from a result entry, used in filtering. If None, defaults to checking `.params` attribute or dict keys.
    get_params_func: Callable | None = None,
    # Optional logger for progress and error messages.
    logger: "Logger" = None,
    # Internal flags for file scanning behavior.
    recursive: bool = True,
    sort_files: bool = True,
    **kwargs,
) -> ResultSet:
    """
    Load lazy entries from a directory (or single file) and apply filters.
    
    This function scans for supported files, extracts parameters from filenames and paths, 
    creates lazy entries, and applies filtering based on provided conditions. 
    
    It also supports optional post-processing of parameters and custom parameter extraction for filtering.
    
    Parameters
    ----------
    data_dir:
        Directory path (or single file) to scan for results.
    filters:
        - None: return all
        - callable: `filters(result) -> bool`
        - dict: `{param: condition}` where condition supports:
          - scalar exact match (numeric with tolerance)
          - list/tuple/set membership
          - tuple operators: ('eq'|'neq'|'lt'|'le'|'gt'|'ge', value)
          - range operator: ('between', (min, max))
          - membership operators: ('in'|'not_in', [v1, ...])
          - string contains: ('contains', 'sub')
          - callable: `lambda param_value, params: ...`
          
    lx, ly, lz, Ns:
        Optional shortcuts for filtering by common size parameters. If provided, they are applied as additional filters on top of `filters`.
    post_process_func:
        Optional function `f(params: dict) -> None` that can modify the extracted parameters in-place. Useful for computing derived parameters or converting units before filtering.
    get_params_func:
        Optional function `f(result) -> dict` to extract parameters from a result entry for filtering. If None, the function will look for a `.params` attribute or use the entry itself if it's a dict.
    logger:
        Optional logger for progress and error messages. Should have methods like `logger.info(msg, color=...)` and `logger.error(msg, color=...)`.
    recursive:
        Whether to scan directories recursively. Default is True.
    sort_files:
        Whether to sort the list of files before processing. Default is True.
    **kwargs:
        Additional keyword arguments for future extensions or specific filtering needs.
        
    Returns
    -------
    ResultSet
        List-like container of lazy entries with convenience methods such as
        ``filtered(...)``, ``show(...)``, and ``show_filtered(...)``.
    """
    path = Path(data_dir)
    results: list[LazyDataEntry] = []

    if not path.exists():
        _log(logger, "error", f"Data path {data_dir} does not exist.", color="red")
        return ResultSet([], get_params_fun=get_params_func, tol=kwargs.get("tol", 1e-9), name=path.name or "results")

    files_iter = _iter_supported_files(path, recursive=recursive)
    if sort_files:
        files_iter = sorted(files_iter)
        _log(logger, "info", f"Found {len(files_iter)} supported files", color="cyan")
    else:
        _log(logger, "info", "Scanning supported files in stream mode (unsorted)", color="cyan")

    for filepath in files_iter:
        try:
            params = _extract_params_from_path(filepath)

            if lx is not None and params.get("Lx") != lx:
                continue
            if ly is not None and params.get("Ly") != ly:
                continue
            if lz is not None and params.get("Lz") != lz:
                continue
            if Ns is not None and params.get("Ns") != Ns:
                continue

            if post_process_func is not None:
                post_process_func(params)

            entry = _entry_from_path(filepath, params)
            if entry is not None:
                results.append(entry)
        except Exception as exc:
            _log(logger, "error", f"Error loading {filepath}: {exc}", color="red")

    result_set = ResultSet(results, get_params_fun=get_params_func, tol=kwargs.get("tol", 1e-9), name=path.name or "results")

    if filters is not None:
        tol = kwargs.pop("tol", 1e-9)
        result_set = filter_results(result_set, filters=filters, get_params_fun=get_params_func, tol=tol)

    _log(logger, "info", f"Loaded {len(result_set)} results after filtering", color="green")
    return result_set

# ----------------------------------------
#! Proxy class for in-memory results that mimics the lazy entry interface.
# ----------------------------------------

class ResultProxy(LazyDataEntry):
    """In-memory result entry with `.params` and dict-like dataset access."""
    __slots__ = ("_data",)

    def __init__(self, data: Union[np.ndarray, dict], params: dict, filepath: str = "<memory>"):
        super().__init__(filepath=filepath, params=params)
        if isinstance(data, dict):
            self._data = dict(data)
        elif isinstance(data, np.ndarray):
            self._data = {"default": data}
        else:
            raise TypeError(f"Data must be dict or ndarray, got {type(data)}")

    def _load_item(self, key: str):
        if key in self._data:
            self._cache[key] = self._data[key]
            return
        raise KeyError(f"Key '{key}' not found in {self.filename}")

    def keys(self):
        return self._data.keys()

    def load_all(self):
        self._cache.update(self._data)
        return self

    def __repr__(self):
        return f"ResultProxy(filepath='{self.filepath}', params={self.params}, keys={list(self._data.keys())})"


def prepare_results_for_plotting(
    data_array          : np.ndarray,
    x_param_values      : list,
    y_param_values      : list,
    *,
    x_param             : str = "J",
    y_param             : str = "hx",
    data_key            : str = "default",
    fixed_params        : dict | None = None,
    post_process_func   : Callable[[dict], None] | None = None,
) -> ResultSet:
    """Convert parameter-grid arrays to ResultProxy list."""
    
    if x_param_values is None or y_param_values is None:
        raise ValueError("x_param_values and y_param_values must be provided for ndarray inputs.")

    fixed_params = {} if fixed_params is None else dict(fixed_params)

    if data_array.shape[0] != len(x_param_values) or data_array.shape[1] != len(y_param_values):
        raise ValueError(
            "data_array first two dimensions must match lengths of x_param_values and y_param_values. "
            f"Got shape={data_array.shape}, len(x)={len(x_param_values)}, len(y)={len(y_param_values)}"
        )

    nx      = len(x_param_values)
    ny      = len(y_param_values)
    results : list[ResultProxy] = [None] * (nx * ny)
    idx     = 0
    for i, x_val in enumerate(x_param_values):
        for j, y_val in enumerate(y_param_values):
            params = {x_param: x_val, y_param: y_val, **fixed_params}
            if post_process_func is not None:
                post_process_func(params)

            data_slice = data_array[i, j, ...]
            results[idx] = ResultProxy(
                data={data_key: data_slice},
                params=params,
                filepath=f"<memory:{x_param}={x_val}_{y_param}={y_val}>",
            )
            idx += 1
    return ResultSet(results, name="memory_results")


class PlotData:
    """Convenience helpers that work with Lazy* entries and ResultProxy."""

    @staticmethod
    def from_input(
        directory       : str | None,
        data_values     : Union[np.ndarray, dict, list, tuple, None] = None,
        x_parameters    : List[float] = None,
        y_parameters    : List[float] = None,
        *,
        x_param         : str = "x",
        y_param         : str = "y",
        data_key        : str = "default",
        filters         : dict | Callable[[Any], bool] | None = None,
        logger          : "Logger" = None,
        **kwargs,
    ) -> ResultSet:
        """Build a plot-ready result list from either filesystem data or in-memory arrays."""
        if directory is not None and data_values is None:
            return load_results(
                data_dir=directory,
                filters=filters,
                lx=kwargs.pop("Lx", kwargs.pop("lx", None)),
                ly=kwargs.pop("Ly", kwargs.pop("ly", None)),
                lz=kwargs.pop("Lz", kwargs.pop("lz", None)),
                Ns=kwargs.pop("Ns", kwargs.pop("ns", None)),
                post_process_func=kwargs.get("post_process_func", None),
                get_params_func=kwargs.get("get_params_func", None),
                logger=logger,
                **kwargs,
            )

        if isinstance(data_values, (list, tuple)):
            return ResultSet(data_values, name="input_results")

        if isinstance(data_values, dict):
            if "params" in data_values and "data" in data_values:
                return ResultSet(
                    [ResultProxy(data=data_values["data"], params=data_values["params"], filepath=data_values.get("filepath", "<memory>"))],
                    name="input_results",
                )

            value_iter = iter(data_values.values())
            first_value = next(value_iter, None)
            if first_value is not None and hasattr(first_value, "params") and all(hasattr(v, "params") for v in value_iter):
                return ResultSet(data_values.values(), name="input_results")

            fixed_params = {k: v for k, v in kwargs.items() if k in {"Lx", "Ly", "Lz", "Ns"}}
            if x_parameters and y_parameters and len(x_parameters) == 1 and len(y_parameters) == 1:
                params = {x_param: x_parameters[0], y_param: y_parameters[0], **fixed_params}
                return ResultSet([ResultProxy(data=data_values, params=params)], name="input_results")
            raise ValueError(
                "For dict data_values provide {'data': ..., 'params': ...} or single-point x/y parameters."
            )

        if isinstance(data_values, np.ndarray):
            fixed_params = {k: v for k, v in kwargs.items() if k in {"Lx", "Ly", "Lz", "Ns"}}
            return prepare_results_for_plotting(
                data_array=data_values,
                x_param_values=x_parameters,
                y_param_values=y_parameters,
                x_param=x_param,
                y_param=y_param,
                data_key=data_key,
                fixed_params=fixed_params,
                post_process_func=kwargs.get("post_process_func", None),
            )

        raise ValueError(
            f"Cannot prepare results from data_values type {type(data_values)}. "
            "Expected directory input, list of results, dict payload, or numpy array."
        )

    @staticmethod
    def from_match(
        results: List[_ENTRY_TYPES],
        x_param: str,
        y_param: str,
        x_val: float,
        y_val: float,
        tol: float = 1e-5,
    ) -> _ENTRY_TYPES | None:
        """Return the first entry matching two parameter values within tolerance."""
        for r in results:
            rx = r.params.get(x_param, np.nan)
            ry = r.params.get(y_param, np.nan)
            if abs(rx - x_val) < tol and abs(ry - y_val) < tol:
                return r
        return None

    @staticmethod
    def extract_parameter_arrays(
        filtered_results: List[_ENTRY_TYPES],
        x_param: str = "J",
        y_param: str = "hx",
        xlim=None,
        ylim=None,
    ):
        """Extract raw and unique numeric x/y parameter arrays."""
        x_vals = np.array([r.params.get(x_param, np.nan) for r in filtered_results], dtype=float)
        y_vals = np.array([r.params.get(y_param, np.nan) for r in filtered_results], dtype=float)

        unique_x = np.unique(x_vals[~np.isnan(x_vals)])
        unique_y = np.unique(y_vals[~np.isnan(y_vals)])

        if xlim is not None:
            lower = -np.inf if xlim[0] is None else xlim[0]
            upper = np.inf if xlim[1] is None else xlim[1]
            unique_x = unique_x[(unique_x >= lower) & (unique_x <= upper)]
        if ylim is not None:
            lower = -np.inf if ylim[0] is None else ylim[0]
            upper = np.inf if ylim[1] is None else ylim[1]
            unique_y = unique_y[(unique_y >= lower) & (unique_y <= upper)]

        return x_vals, y_vals, unique_x, unique_y

    @staticmethod
    def sort_results_by_param(results: List[_ENTRY_TYPES], param_name: str):
        vals = np.array([r.params.get(param_name, np.nan) for r in results], dtype=float)
        sort_idx = np.argsort(vals)
        return vals[sort_idx], sort_idx

    @staticmethod
    def determine_vmax_vmin(
        results: List[_ENTRY_TYPES],
        param_name: str,
        param_fun: Callable = lambda r, name: r.params[name],
        nstates: int = None,
    ):
        all_values = []
        for r in results:
            try:
                values = param_fun(r, param_name)
                if isinstance(values, (list, np.ndarray)):
                    vals = np.array(values[:nstates]).flatten() if nstates is not None else np.array(values).flatten()
                    all_values.extend(np.real(vals).tolist())
                else:
                    all_values.append(np.real(values))
            except Exception:
                pass

        if not all_values:
            return np.nan, np.nan

        arr = np.array(all_values, dtype=float).flatten()
        return np.nanmin(arr), np.nanmax(arr)

    @staticmethod
    def savefig(
        fig,
        directory: str,
        *name_parts,
        suffix: str = "",
        ext: str = "png",
        dpi: int = 250,
        logger: "Logger" = None,
    ) -> Path:
        """Save figure to `directory` with a generated file name."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_parts = [str(p).strip().replace("/", "_") for p in name_parts if p is not None and str(p).strip() != ""]
        stem = "_".join(safe_parts) if safe_parts else "figure"
        if suffix:
            stem = f"{stem}{suffix}"

        out_path = out_dir / f"{stem}.{ext.lstrip('.') }"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        _log(logger, "info", f"Saved figure: {out_path}", color="green")
        return out_path


__all__ = [
    "parse_filename",
    "ResultSet",
    "filter_results",
    "load_results",
    "ResultProxy",
    "prepare_results_for_plotting",
    "PlotData",
]

# ------------------------
#! EOF
# ------------------------
