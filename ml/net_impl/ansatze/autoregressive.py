"""
Compatibility export for the autoregressive ansatz.

This is reusable as a general-purpose autoregressive network, but it is currently only
used as an ansatz in NQS. It is implemented as a thin wrapper around the more general
``ComplexAR`` in ``QES.general_python.ml.net_impl.networks.net_autoregressive``, which can be used directly outside NQS.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
--------------------------------
"""

from ..networks.net_autoregressive import ComplexAR

__all__ = ["ComplexAR"]
