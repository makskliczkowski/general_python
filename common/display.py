"""

file    : general_python/common/display.py
desc    : Helpers for rendering quantum-mechanical objects in a Jupyter notebook using
            LaTeX via IPython.display.Math.

Helpers for rendering quantum-mechanical objects in a Jupyter notebook using
LaTeX via IPython.display.Math.

Functions are intentionally small and composable so they chain well inside a
notebook cell.  All helper names start with `display_…` to make tab-completion
useful.

The module does **not** depend on your project-specific `bin_mod`; if you have a
custom integer→binary converter just pass it as the optional `to_bin`
parameter.
"""

from typing import Iterable, Callable, List, Sequence, Tuple, Union
from IPython.display import display, Math
import numpy as np

# ---------------------------------------------------------------------------
#! Thin util: convert an integer Fock state to a binary string
# ---------------------------------------------------------------------------

def _isinteger(state: int | np.ndarray) -> bool:
    return isinstance(state, (int, np.integer, np.int32, np.int64))

def _default_to_bin(state: int, ns: int) -> str:
    """Return the *ns*-bit binary representation of *state* as a string."""
    if isinstance(state, list):
        # replace -1s with 0s
        statex      = [0 if s == -1 else s for s in state]
        strtex      = str(statex)[1:-1].replace('.', '')
        strtex      = strtex.replace(' ', '').replace('-1', '0')        
        return strtex
    elif _isinteger(state):
        return format(state, f"0{ns}b")
    
    # replace -1s with 0s
    statex      = np.where(state == -1, 0, state)
    strtex      = str(statex)[1:-1].replace('.', '')
    strtex      = strtex.replace(' ', '').replace('-1', '0')        
    return strtex

def _format_coeff(c: complex | float) -> str:
    """Return LaTeX for a numeric coefficient with phase/amplitude split."""
    if isinstance(c, complex):
        amp   = abs(c)
        phase = np.angle(c)
        if amp == 0:
            return "0"
        if amp == 1:
            return fr"e^{{i\,{phase:.3g}}}" if phase != 0 else ""
        else:
            return fr"{amp:g}e^{{i\,{phase:.3g}}}" if phase != 0 else f"{amp:g}"
    elif isinstance(c, float):
        if c in (1, -1):
            return "" if c == 1 else "-"
        return f"{c:g}"
    elif _isinteger(c):
        if c in (1, -1):
            return "" if c == 1 else "-"
        return str(c)
    else:
        return str(c)

# ---------------------------------------------------------------------------
#! |ψ⟩  and  ⟨ψ|  helpers
# ---------------------------------------------------------------------------

def ket(state: int, ns: int, *, to_bin: Callable[[int, int], str] | None = None) -> str:
    r"""LaTeX code for a computational-basis ket |(int) b...⟩.

    Example
    -------
    >>> ket(6, 4)
    '|(6)\,0101\rangle'
    """
    to_bin = _default_to_bin if (to_bin is None) else to_bin
    if _isinteger(state):
        return fr"\left|({state})\,{to_bin(state, ns)}\right\rangle"
    return fr"\left|{to_bin((state + 1) // 2, ns)}\right\rangle"


def bra(state: int, ns: int, *, to_bin: Callable[[int, int], str] | None = None) -> str:
    r"""LaTeX code for the bra ⟨ψ| corresponding to :pyfunc:`ket`."""
    return ket(state, ns, to_bin=to_bin).replace("\\left|", "\\left\langle").replace("\\right\\rangle", "\\right|")

# ---------------------------------------------------------------------------
#! Display primitives
# ---------------------------------------------------------------------------

def display_state(state : Union[int, np.ndarray, List[int]],
                ns      : int,
                *,
                label   : str | None = None,
                to_bin  : Callable[[int, int], str] | None = None) -> None:
    """Render a single basis state.

    Parameters
    ----------
    state, ns
        Integer basis state and number of sites.
    label
        Optional text (e.g. *Initial*) shown before the ket.
    """
    if _isinteger(state):
        latex       = ket(state, ns, to_bin=to_bin)
    else:
        statestr    = str(state)
        latex       = fr"|{statestr[1:-1].replace('.', '')}\rangle"
        
    if label:
        latex = fr"\text{{{label}: }}\,{latex}"
    display(Math(latex))

def display_operator_action(op_tex    : str,
                            site      : int | tuple[int, ...],
                            in_state  : Union[int, np.ndarray, List[int]],
                            ns        : int,
                            out_state : int | Sequence[int] | None,
                            coeff     : float | complex | Sequence[float | complex],
                            *,
                            to_bin    : Callable[[int, int], str] | None = None
                            ) -> None:
    r"""
    Show the action  Ô₍site₎ |ψ⟩ = Σₖ cₖ |φₖ⟩  (or 0).

    Parameters
    ----------
    op_tex : str
        LaTeX operator label without subscript, e.g. ``'c^\\dagger'``.
    site : int | tuple[int,…]
        Site index (or indices) → subscript in Ô₍site₎.
    in_state : int | ndarray | list[int]
        Input basis state or wave-function label shown on the LHS.
    ns : int
        Number of lattice sites (binary string length for integer states).
    out_state : int | list[int] | None
        Output basis state(s).  *None* ⇒ result is 0.
    coeff : number | complex | list[number|complex]
        Corresponding amplitude(s).  Must match *out_state* length.
    """
    to_bin = _default_to_bin if to_bin is None else to_bin

    # left-hand side
    sub   = ",".join(map(str, site)) if isinstance(site, tuple) else str(site)
    if _isinteger(in_state):
        lhs = fr"{op_tex}_{{{sub}}}\,{ket(in_state, ns, to_bin=to_bin)}"
    else:
        statestr = fr"|{str(in_state)[1:-1].replace('.', '')}\rangle"
        lhs      = fr"{op_tex}_{{{sub}}}\,{statestr}"

    # right-hand side
    if out_state is None:
        rhs = "0"
    else:
        # ensure lists
        out_list    = list(out_state) if isinstance(out_state, Sequence) else [out_state]
        coeff_list  = list(coeff)     if isinstance(coeff, Sequence)     else [coeff]

        if len(out_list) != len(coeff_list):
            raise ValueError("out_state and coeff must have equal length")

        # build Σ_k c_k |φ_k⟩
        terms = []
        for c, s in zip(coeff_list, out_list):
            if c == 0:            # skip zero amplitudes
                continue
            coeff_tex = _format_coeff(c)
            ket_tex   = ket(s, ns, to_bin=to_bin)
            term      = coeff_tex + ket_tex
            terms.append(term)

        rhs = " + ".join(terms) if terms else "0"

    # display
    display(Math(lhs + " = " + rhs))

# ---------------------------------------------------------------------------
#! Superpositions and products
# ---------------------------------------------------------------------------

def superposition(terms : Sequence[Tuple[complex, int]],
                ns      : int,
                *, 
                to_bin  : Callable[[int, int], str] | None = None) -> str:
    """Return LaTeX for Σₖ aₖ |k⟩ given *(coeff, state)* pairs."""
    parts: List[str] = []
    to_bin = _default_to_bin if to_bin is None else to_bin
    for amp, st in terms:
        amp_tex = f"{amp:g}" if amp not in (-1, 1) else ("-" if amp == -1 else "")
        parts.append(fr"{amp_tex}{ket(st, ns, to_bin=to_bin)}")
    return " + ".join(parts) or "0"

def display_superposition(terms : Sequence[Tuple[complex, int]],
                        ns      : int,
                        *, 
                        label   : str | None = None,
                        to_bin  : Callable[[int, int], str] | None = None) -> None:
    latex = superposition(terms, ns, to_bin=to_bin)
    if label:
        latex = fr"\text{{{label}: }}\,{latex}"
    display(Math(latex))

# ---------------------------------------------------------------------------
#! Examples as doctests (run them in a notebook to verify)  --------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ns = 4
    display_state(3, ns, label="Initial")
    new = 3 ^ (1 << (ns - 1 - 2))   # flip site 2
    display_operator_action("c^\\dagger", 2, 3, ns, new, -1)
    display_superposition([(0.5, 0), (0.5, 15)], ns, label="GHZ")
