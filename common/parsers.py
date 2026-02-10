'''
Parsers and string utilities for general use.

-------------------------
Author : Maksymilian Kliczkowski
Date   : 2026-01-15
-------------------------
'''
from    __future__  import annotations

from    dataclasses import dataclass
from    typing      import Callable, Tuple, Optional, Sequence, Any, List, Union, Dict, Type, Union, TYPE_CHECKING
import  re

import  os
import  sys
import  traceback

import  pandas as pd
from    warnings import simplefilter

if TYPE_CHECKING:
    from .flog import Logger

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

Number  = Union[int, float]
Context = dict[str, int]

# ─────────────────────────────────────────────────────────────────────────────
# suppress pandas PerformanceWarning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# ─────────────────────────────────────────────────────────────────────────────
def filter_dataframe(df     : pd.DataFrame,
                criteria    : Dict[str, Sequence[Any]],
                *,
                copy        : bool = False) -> pd.DataFrame:
    r"""
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
    def log(err: Exception, msg: str, *skip_types: Type[BaseException], logger: 'Logger' = None ) -> None:
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

        if logger is None:
            print(f"Error in {filename} at line {lineno}")
            print(f"Message: {msg}")
            print(f"Exception: {err}")
            print(traceback.format_exc())
        else:
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
        r"""
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
#! Command line argument parsing utilities
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParseSpec:
    """
    A spec describing how to interpret a token that references a base via a suffix/prefix.

    Supported token forms (examples):
      - prefix op:    "^0.5"  or "*2.0"   -> base_key op value
      - suffix op:    "0.5^"  or "2.0*"   -> base_key_suffix op value

    where base_key and base_key_suffix may differ (e.g. ns vs nh).
    """
    op_char         : str       # '^' or '*'
    base_key_prefix : str       # which base to use for prefix form, e.g. "ns"
    base_key_suffix : str       # which base to use for suffix form, e.g. "nh"

def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception as e:
        raise ValueError(f"Cannot parse number from '{s}'.") from e
    

def parse_int_list(
    s: str,
    *,
    ctx: Context,
    max_key: str,
    specs: Sequence[ParseSpec],
    default_ctx: Optional[dict[str, Callable[[Context], int]]] = None,
    sep: str = ",",
    allow_ranges: bool = True,
    range_step_sep: str = ":",
    allow_duplicates: bool = False,
    sort_unique: bool = False,
    clamp: bool = False,
) -> List[int]:
    """
    General CLI-style parser that produces a list of positive integers in [1, ctx[max_key]].

    Token grammar (comma-separated by default):
      1) Integer / float >= 1:
           "32"   -> 32
           "32.0" -> 32

      2) Base exponentiation:
           "^p"   -> int(ctx[spec.base_key_prefix] ** p)   for op_char='^'
           "p^"   -> int(ctx[spec.base_key_suffix] ** p)

      3) Base multiplication:
           "*a"   -> int(ctx[spec.base_key_prefix] * a)    for op_char='*'
           "a*"   -> int(ctx[spec.base_key_suffix] * a)

      4) Ranges (optional):
           "a-b"         -> a, a+1, ..., b
           "a-b:step"    -> a, a+step, ..., <= b

    Notes:
      - ctx is a dictionary of named integer bases, e.g. {"ns": ns, "nh": 2**ns}.
      - max_key names the variable providing the upper bound, typically "nh".
      - default_ctx optionally computes missing ctx keys from existing ones.
      - clamp=True clamps out-of-range values into [1, max], otherwise rejects them.
    """
    
    if not isinstance(s, str):
        raise TypeError(f"Expected a string, got {type(s).__name__}.")

    if default_ctx is not None:
        for k, fn in default_ctx.items():
            if k not in ctx:
                ctx[k] = int(fn(ctx))

    if max_key not in ctx:
        raise ValueError(f"ctx must contain '{max_key}' for bounds checking.")
    
    max_val = int(ctx[max_key])
    if max_val < 1:
        raise ValueError(f"ctx['{max_key}'] must be >= 1, got {max_val}.")

    spec_map = {sp.op_char: sp for sp in specs}

    def accept(v: int, out: List[int]) -> None:
        if clamp:
            v = max(1, min(max_val, v))
        if not (1 <= v <= max_val):
            raise ValueError(f"Value {v} out of bounds [1, {max_val}].")
        out.append(v)

    out: List[int]  = []
    tokens          = [t.strip() for t in s.split(sep) if t.strip() != ""]
    # Range regex
    range_re        = re.compile(r"^\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*(?::\s*([+-]?\d+)\s*)?$")

    for tok in tokens:
        # Ranges: "a-b" or "a-b:step"
        if allow_ranges:
            m = range_re.match(tok)
            if m is not None:
                a    = int(m.group(1))
                b    = int(m.group(2))
                step = int(m.group(3)) if m.group(3) is not None else 1
                if step == 0:
                    raise ValueError(f"Invalid range step 0 in '{tok}'.")
                if (b - a) * step < 0:
                    raise ValueError(f"Range step sign inconsistent in '{tok}'.")
                v = a
                if step > 0:
                    while v <= b:
                        accept(v, out)
                        v += step
                else:
                    while v >= b:
                        accept(v, out)
                        v += step
                continue

        # Operator-based forms: '^' or '*'
        op_used = None
        for op in spec_map:
            if op in tok:
                op_used = op
                break

        if op_used is not None:
            sp      = spec_map[op_used]
            parts   = tok.split(op_used)
            if len(parts) != 2:
                raise ValueError(f"Malformed token '{tok}' (too many '{op_used}').")

            left, right = parts[0].strip(), parts[1].strip()

            # prefix form: "^p" or "*a"
            if left == "" and right != "":
                x         = _to_float(right)
                base_key  = sp.base_key_prefix
                if base_key not in ctx:
                    raise ValueError(f"Missing ctx['{base_key}'] for token '{tok}'.")
                base      = float(ctx[base_key])
                v         = int(base ** x) if op_used == "^" else int(base * x)
                accept(v, out)
                continue

            # suffix form: "p^" or "a*"
            if right == "" and left != "":
                x         = _to_float(left)
                base_key  = sp.base_key_suffix
                if base_key not in ctx:
                    raise ValueError(f"Missing ctx['{base_key}'] for token '{tok}'.")
                base      = float(ctx[base_key])
                v         = int(base ** x) if op_used == "^" else int(base * x)
                accept(v, out)
                continue

            raise ValueError(f"Malformed token '{tok}'. Use '{op_used}<x>' or '<x>{op_used}'.")

        # Plain number token
        x = _to_float(tok)
        if x < 1:
            raise ValueError(f"Value must be >= 1, got {x} in '{tok}'.")
        accept(int(x), out)

    if not allow_duplicates:
        out = list(dict.fromkeys(out))  # preserves order

    if sort_unique:
        out = sorted(set(out))

    return out


# ─────────────────────────────────────────────────────────────────────────────
#! END OF FILE
# ─────────────────────────────────────────────────────────────────────────────