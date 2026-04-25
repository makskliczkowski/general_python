"""
Compatibility export for the autoregressive ansatz.

This is reusable as a general-purpose autoregressive network. It is implemented
as a thin wrapper around ``ComplexAR`` in
``QES.general_python.ml.net_impl.networks.net_autoregressive``.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
--------------------------------
"""

from ..networks.net_autoregressive import ComplexAR

__all__ = ["ComplexAR"]
