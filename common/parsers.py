import os
import sys
import traceback
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import pandas as pd
from warnings import simplefilter
from ..common.flog import get_global_logger

# ─────────────────────────────────────────────────────────────────────────────
# suppress pandas PerformanceWarning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# configure module‐level logger
logger = get_global_logger()

# ─────────────────────────────────────────────────────────────────────────────
def filter_dataframe(df     : pd.DataFrame,
                criteria    : Dict[str, Sequence[Any]],
                *,
                copy        : bool = False) -> pd.DataFrame:
    """
    Return `df` filtered to rows where each column key \in criteria contains one of the allowed values.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    criteria : dict of {column_name: allowed_values}
        Each column is filtered via `df[column_name].isin(allowed_values)`.
    copy : bool, default=False
        If True, operate on `df.copy()`, otherwise filter in-place.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    target = df.copy() if copy else df
    for col, vals in criteria.items():
        target = target[target[col].isin(vals)]
    return target

# ─────────────────────────────────────────────────────────────────────────────
class ExceptionHandler:
    """
    Centralized exception logging.

    Usage
    -----
    try:
        ...
    except Exception as err:
        ExceptionHandler.log(err, "context message", ValueError, KeyError)
    """
    @staticmethod
    def log(err: Exception, msg: str, *skip_types: Type[BaseException]) -> None:
        """
        Log exception info unless its type is in `skip_types`.

        Parameters
        ----------
        err : Exception
            Caught exception.
        msg : str
            Contextual message.
        skip_types : Exception types
            If `err` is instance of any, logging is skipped.
        """
        if any(isinstance(err, t) for t in skip_types):
            return

        exc_type, _, tb = sys.exc_info()
        frame_info      = tb.tb_frame if tb else None
        filename        = os.path.basename(frame_info.f_code.co_filename) if frame_info else "<unknown>"
        lineno          = tb.tb_lineno if tb else "?"

        logger.error("Error in %s at line %s", filename, lineno)
        logger.error("Message: %s", msg)
        logger.error("Exception: %s", err)
        logger.error(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# String‐parsing conventions
_DIV        = "_"
_CORR       = "-"
_MULT       = ","
_RANGE      = "."
_SEP        = "/"

def approx_equal(val1: Union[str, float], val2: Union[str, float], precision: int = 3) -> bool:
    """
    Compare two numeric values up to given decimal places.

    Parameters
    ----------
    val1, val2 : str or float
        Values to compare.
    precision : int
        Number of decimal places.

    Returns
    -------
    bool
        True if rounded(val1, precision) == rounded(val2, precision).
    """
    return round(float(val1), precision) == round(float(val2), precision)

class StringParser:
    """
    Utilities for formatting and splitting parameterized strings.
    """

    @staticmethod
    def sci(x: float, prec: int) -> str:
        """
        Format `x` in scientific notation with `prec` decimals:  
        $$ x \\approx d.ddd\\times10^{\\pm e} $$
        """
        s = format(x, f".{prec}E")
        # normalize exponent sign
        return s.replace("E+0", "E+").replace("E-0", "E-")

    @staticmethod
    def join_list(items: Sequence[Any], elem_fmt: Callable[[Any], str] = str, sep: str = ",", bracketed: bool = True) -> str:
        """
        Join a list of elements into a string.

        Parameters
        ----------
        items : sequence of values
        elem_fmt : callable
            Applied to each item before joining.
        sep : str
            Separator between elements.
        bracketed : bool
            If True, wrap result in [ ].
        """
        body = sep.join(elem_fmt(it) for it in items)
        return f"[{body}]" if bracketed else body

    @staticmethod
    def parse_filename_param(filename: str, section: int, param_idx: int) -> Tuple[str, float]:
        """
        From a filename like '..._key1=val1,key2=val2_...', extract
        the (key, value) pair in the specified section.

        Parameters
        ----------
        filename : str
            Full filename to parse.
        section : int
            Index after splitting by `_`.
        param_idx : int
            Index of comma-separated `key=value` pair in that section.

        Returns
        -------
        (key, value)
        """
        part = filename.split(_DIV)[section]
        kv   = part.split(_MULT)[param_idx].split("=")
        return kv[0], float(kv[1])

# ─────────────────────────────────────────────────────────────────────────────
#! END OF FILE
# ─────────────────────────────────────────────────────────────────────────────