"""
Compatibility export for the group-equivariant GCNN ansatz.

This is reusable as a general-purpose symmetry-aware GCNN. It is implemented as
a thin wrapper around ``GCNN`` in
``QES.general_python.ml.net_impl.networks.net_gcnn``.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
--------------------------------
"""

from ..networks.net_gcnn import EquivariantGCNN

__all__ = ["EquivariantGCNN"]
