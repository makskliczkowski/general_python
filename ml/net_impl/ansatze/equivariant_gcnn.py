"""
Compatibility export for the group-equivariant GCNN ansatz.

This is reusable as a general-purpose GCNN, but it is currently only used as an ansatz in NQS. It is implemented as a thin wrapper around the more general
``GCNN`` in ``QES.general_python.ml.net_impl.networks.net_gcnn``, which can be used directly outside NQS.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
--------------------------------
"""

from ..networks.net_gcnn import EquivariantGCNN

__all__ = ["EquivariantGCNN"]
