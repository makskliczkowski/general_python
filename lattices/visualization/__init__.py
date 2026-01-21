"""
Utilities for presenting lattice data in human-friendly forms.

This subpackage groups together text formatters and plotting helpers
intended to work with :class:`~general_python.lattices.lattice.Lattice`
instances.  The helpers are designed to be lightweight wrappers that:

* avoid mutating the lattice objects,
* operate on NumPy arrays derived from lattice state,
* return plain Python data structures (strings, matplotlib figures)
  so that callers can further customise behaviour.
"""

from .formatting import (
    format_lattice_summary,
    format_vector_table,
    format_real_space_vectors,
    format_reciprocal_space_vectors,
    format_brillouin_zone_overview,
)
from .plotting import (
    LatticePlotter,
    plot_real_space,
    plot_reciprocal_space,
    plot_brillouin_zone,
    plot_lattice_structure,
)

__all__ = [
    "format_lattice_summary",
    "format_vector_table",
    "format_real_space_vectors",
    "format_reciprocal_space_vectors",
    "format_brillouin_zone_overview",
    "LatticePlotter",
    "plot_real_space",
    "plot_reciprocal_space",
    "plot_brillouin_zone",
    "plot_lattice_structure",
]
