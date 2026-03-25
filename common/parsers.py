"""
Parsers and string utilities for general use.

-------------------------
Author  : Maksymilian Kliczkowski
Date    : 2026-01-15
Version : 2.0
-------------------------
"""
from    __future__  import annotations

from    dataclasses import dataclass
from    typing      import Callable, Tuple, Optional, Sequence, Any, List, Union, Dict, Type, Union, TYPE_CHECKING
import  re
import  math

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
Context = Dict[str, Union[int, float]]

# ─────────────────────────────────────────────────────────────────────────────
# suppress pandas PerformanceWarning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_expression(tok: str, ctx: Context) -> float:
    """
    Evaluate one token as a mathematical expression in the supplied context.

    The token is first normalized with `_normalize_math_expression`, then
    evaluated with a restricted namespace containing:

    - numeric names from `ctx`, for example `ns`, `nh`, `N`, `L`
    - selected Python builtins: `abs`, `min`, `max`, `round`, `int`, `float`, `pow`
    - public functions/constants from `math`, for example `sqrt`, `pi`, `sin`

    This helper is intended as the final fallback in `parse_int_list` after
    simpler token forms such as plain numbers, ranges, or prefix/suffix specs
    have already been checked.

    Examples
    --------
    >>> _evaluate_expression("N/2", {"N": 8})
    4.0
    >>> _evaluate_expression("L^2", {"L": 6})
    36.0
    >>> _evaluate_expression("max(ns, nh/10)", {"ns": 8, "nh": 70})
    8.0

    Step by step for ``"L^2"``:
    1. `_normalize_math_expression("L^2")` converts it to ``"L**2"``.
    2. `ctx["L"]` is injected into the safe namespace.
    3. The normalized expression is evaluated and converted to `float`.
    """
    expr        = _normalize_math_expression(tok)
    
    # Safe namespace construction
    safe_dict   = {
        'abs': abs, 'min': min, 'max': max, 'round': round,
        'int': int, 'float': float, 'pow': pow,
    }
    
    # Add math functions
    for name in dir(math):
        if not name.startswith('_'):
            safe_dict[name] = getattr(math, name)
            
    # Add everything from ctx
    safe_dict.update(ctx)
    
    # Eval with no builtins for security
    try:
        return float(eval(expr, {"__builtins__": {}}, safe_dict))
    except Exception as e:
        raise ValueError(f"Expression evaluation failed for '{tok}': {e}")

def _normalize_math_expression(expr: str) -> str:
    """
    Normalize common mathematical Unicode/operator variants into Python syntax.

    This keeps CLI/config expressions readable while still evaluating them
    through the same safe expression path.

    Conversions currently include:

    - `^ -> **`
    - `×, · -> *`
    - `÷ -> /`
    - common Unicode minus variants to ASCII `-`
    - full-width `+`, `-`, `*`, `/` to ASCII operators

    Examples
    --------
    >>> _normalize_math_expression("N^2")
    'N**2'
    >>> _normalize_math_expression("L×L − 1")
    'L*L - 1'
    >>> _normalize_math_expression("Dq÷10")
    'Dq/10'
    """
    return (
        expr
        .replace("^", "**")
        .replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
        .replace("－", "-")
        .replace("＊", "*")
        .replace("／", "/")
    )

def _normalize_operator_token(expr: str) -> str:
    """
    Normalize operator glyph variants while preserving parser control chars
    like '^', '*', 'x', and 'v' for prefix/suffix token handling.

    This helper is narrower than `_normalize_math_expression`: it deliberately
    does not rewrite `^` into `**`, because tokens like `^0.5` and `2^`
    have special meaning for `parse_int_list`.

    Examples
    --------
    >>> _normalize_operator_token("L×L")
    'L*L'
    >>> _normalize_operator_token("N−1")
    'N-1'
    >>> _normalize_operator_token("^0.5")
    '^0.5'
    """
    return (
        expr
        .replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
        .replace("－", "-")
        .replace("＊", "*")
        .replace("／", "/")
    )

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
    Small utilities for formatting and parsing parameterized strings.

    The methods here are intentionally simple and deterministic. They are used
    for filename fragments and display formatting rather than for full CLI math
    expressions. For expression-like command-line parsing, see `parse_int_list`.
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

        Examples
        --------
        >>> StringParser.parse_filename_param("run_ns=12,gamma=5_seed=1.h5", 1, 0)
        ('ns', 12.0)
        >>> StringParser.parse_filename_param("run_ns=12,gamma=5_seed=1.h5", 1, 1)
        ('gamma', 5.0)

        Step by step for ``section=1, param_idx=1``:
        1. Split the filename by `_`.
        2. Take section `1`, here ``"ns=12,gamma=5"``.
        3. Split that section by `,`.
        4. Take parameter `1`, here ``"gamma=5"``.
        5. Split by `=` and convert the value to `float`.
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

    Examples
    --------
    >>> spec = ParseSpec(op_char="^", base_key_prefix="ns", base_key_suffix="nh")
    >>> spec.op_char
    '^'

    With `ctx = {"ns": 8, "nh": 70}` this spec means:
    - `^2`   -> `int(8 ** 2)`  -> `64`
    - `2^`   -> `int(70 ** 2)` -> `4900`

    The actual bounds check is performed later by `parse_int_list`.
    """
    op_char         : str       # '^' or '*'
    base_key_prefix : str       # which base to use for prefix form, e.g. "ns"
    base_key_suffix : str       # which base to use for suffix form, e.g. "nh"

def _to_float(s: str) -> float:
    """
    Convert a token to `float` and raise a parser-friendly error on failure.

    Examples
    --------
    >>> _to_float("3")
    3.0
    >>> _to_float("0.25")
    0.25
    """
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

    This function is meant for CLI/config inputs such as:

    - `"1,2,5"`
    - `"1-10:2"`
    - `"^0.5,*2"`
    - `"N/2,N-1,L*L,Dq/10"`

    Parsing order
    -------------
    Each comma-separated token is processed in this order:

    1. Range parsing, if enabled:
       - `"3-6"`      -> `3, 4, 5, 6`
       - `"3-9:2"`    -> `3, 5, 7, 9`
    2. Prefix/suffix operator specs from `specs`:
       - `^0.5` with base `ns=16` -> `int(16 ** 0.5)` -> `4`
       - `2*` with base `nh=70`   -> `int(70 * 2)`    -> `140`
    3. Plain numeric token:
       - `"32"` or `"32.0"` -> `32`
    4. General expression fallback with `_evaluate_expression`:
       - `"N/2"`   -> `int(ctx["N"] / 2)`
       - `"L*L"`   -> `int(ctx["L"] * ctx["L"])`
       - `"max(L,N)"` -> `int(max(ctx["L"], ctx["N"]))`

    Bounds and deduplication
    ------------------------
    Every accepted integer is validated against `[1, ctx[max_key]]`.

    - If `clamp=False`, out-of-range values raise `ValueError`.
    - If `clamp=True`, out-of-range values are clipped into the valid interval.
    - If `allow_duplicates=False`, duplicates are removed while preserving
      first appearance order.
    - If `sort_unique=True`, the final output becomes a sorted unique list.

    Ambiguities
    -----------
    Purely numeric forms of the shape `"a-b"` are reserved for ranges, not
    subtraction. This means:

    - `"1-5"` -> range expansion `[1, 2, 3, 4, 5]`
    - `"N-1"` -> subtraction expression `ctx["N"] - 1`
    - `"L-2"` -> subtraction expression `ctx["L"] - 2`

    In practice, symbolic left-hand sides such as `N`, `L`, `D`, or `Dq`
    are safe and do not collide with the numeric range parser.

    Parameters
    ----------
    s : str
        Input string containing tokens separated by `sep`.
    ctx : dict
        Context of named numeric values used by special forms and expressions,
        for example `{"ns": 16, "nh": 70, "N": 4, "L": 8}`.
    max_key : str
        Name of the context entry that defines the allowed upper bound.
    specs : sequence of ParseSpec
        Prefix/suffix token rules. Typical examples are `^`, `*`, `x`, `v`.
    default_ctx : dict[str, callable], optional
        Lazy context fillers. If a key is missing or set to `None`, the callable
        is evaluated and stored into `ctx`.
    sep : str, default=","
        Token separator.
    allow_ranges : bool, default=True
        Whether forms like `"a-b"` and `"a-b:step"` are accepted.
    range_step_sep : str, default=":"
        Reserved range step separator. The current range parser expects `:`.
    allow_duplicates : bool, default=False
        Whether repeated values should be preserved.
    sort_unique : bool, default=False
        Whether to replace the output with `sorted(set(out))` at the end.
    clamp : bool, default=False
        Whether to clamp values into `[1, ctx[max_key]]` instead of rejecting them.

    Returns
    -------
    list[int]
        Parsed integer values.

    Examples
    --------
    Basic numbers:
    >>> parse_int_list("1,2,5", ctx={"nh": 10}, max_key="nh", specs=[])
    [1, 2, 5]

    Range expansion:
    >>> parse_int_list("2-6:2", ctx={"nh": 10}, max_key="nh", specs=[])
    [2, 4, 6]

    Prefix/suffix parsing:
    >>> specs = [ParseSpec("^", "ns", "nh"), ParseSpec("*", "ns", "nh")]
    >>> parse_int_list("^2,*0.5", ctx={"ns": 8, "nh": 100}, max_key="nh", specs=specs)
    [64, 4]

    Expression fallback:
    >>> parse_int_list("N/2,L*L,max(L,N)", ctx={"N": 4, "L": 8, "nh": 100}, max_key="nh", specs=[])
    [2, 64, 8]

    Unicode operators are normalized:
    >>> parse_int_list("L×L,N−1,D÷10", ctx={"L": 8, "N": 4, "D": 70, "nh": 100}, max_key="nh", specs=[])
    [64, 3, 7]

    Step-by-step example
    --------------------
    Suppose:

    >>> ctx = {"ns": 8, "nh": 70, "N": 4, "L": 8}
    >>> specs = [ParseSpec("^", "ns", "nh"), ParseSpec("*", "ns", "nh")]

    Then parsing ``"^2,N/2,3-5"`` proceeds as:

    1. `^2`
       - matched by the `^` spec as a prefix token
       - uses `ctx["ns"] = 8`
       - computes `int(8 ** 2) = 64`
    2. `N/2`
       - not a range, not a prefix/suffix spec
       - evaluated as an expression with `ctx["N"] = 4`
       - computes `int(4 / 2) = 2`
    3. `3-5`
       - matched as a range
       - expands to `3, 4, 5`

    Final result:

    >>> parse_int_list("^2,N/2,3-5", ctx=ctx, max_key="nh", specs=specs)
    [64, 2, 3, 4, 5]
    """
    
    if not isinstance(s, str):
        raise TypeError(f"Expected a string, got {type(s).__name__}.")

    if default_ctx is not None:
        for k, fn in default_ctx.items():
            if k not in ctx or ctx[k] is None:
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

        # Operator-based forms (prefix/suffix only)
        matched_spec    = False
        tok_ops         = _normalize_operator_token(tok)
        for op, sp in spec_map.items():
            if op in tok_ops:
                parts   = tok_ops.split(op)
                if len(parts) == 2:
                    left, right = parts[0].strip(), parts[1].strip()
                    # prefix form: "^p" or "*a"
                    if left == "" and right != "":
                        x               = _to_float(right)
                        base_key        = sp.base_key_prefix
                        if base_key not in ctx:
                            raise ValueError(f"Missing ctx['{base_key}'] for token '{tok}'.")
                        base            = float(ctx[base_key])
                        v               = int(base ** x) if op == "^" else int(base * x)
                        matched_spec    = True
                        accept(v, out)
                        break
                    
                    # suffix form: "p^" or "a*"
                    elif right == "" and left != "":
                        x               = _to_float(left)
                        base_key        = sp.base_key_suffix
                        if base_key not in ctx:
                            raise ValueError(f"Missing ctx['{base_key}'] for token '{tok}'.")
                        base            = float(ctx[base_key])
                        v               = int(base ** x) if op == "^" else int(base * x)
                        matched_spec    = True
                        accept(v, out)
                        break
        if matched_spec:
            continue

        # Plain number token or expression evaluation
        try:
            x = _to_float(tok)
            if x < 1:
                raise ValueError(f"Value must be >= 1, got {x} in '{tok}'.")
            accept(int(x), out)
        except ValueError:
            # Fallback: try evaluating as expression using ctx
            try:
                val = _evaluate_expression(tok, ctx)
                if val < 1:
                    raise ValueError(f"Value evaluated to {val}, must be >= 1 in '{tok}'.")
                accept(int(val), out)
            except Exception as e:
                raise ValueError(f"Cannot parse token '{tok}' from '{s}'. Reason: {e}")

    if not allow_duplicates:
        out = list(dict.fromkeys(out))  # preserves order

    if sort_unique:
        out = sorted(set(out))

    return out


# ─────────────────────────────────────────────────────────────────────────────
#! END OF FILE
# ─────────────────────────────────────────────────────────────────────────────
