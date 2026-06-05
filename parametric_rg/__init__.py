"""parametric_rg: a Python/Sage port of Fakouri-Rahmany-Basiri (2018)
parametric Rosenfeld-Groebner.

Import inside a Sage session (the (N,W) parametric side needs sage.all)::

    from parametric_rg.main import PRG
    from parametric_rg.ring import build_ring

See README.md and ~/project/reports/parametric-rg-port.md.
"""
from .diffprim import DeltaPolynomial          # noqa: F401
