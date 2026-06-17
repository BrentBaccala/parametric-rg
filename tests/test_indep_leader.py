"""Regression test: independent-variable (coordinate-relation) leaders.

A coordinate relation such as ``x*y - z`` has an independent variable as its
LeadingDerivative. deg_in / isreduceble / leaderreduced used to crash on these:
sympy's ``Poly(p, x)`` raises PolynomialError when p contains a jet of x (e.g.
``rho(x,y,z)``). The original par-rga also mishandles them -- Maple's
``FactorDerivative(x)`` raises 'dependent variable expected' -- but the
coefficient swell halts it long before such vertices arise (this surfaced only
once a per-vertex diffprem wall got the port past the swell).

The port is now robust: ``deg_in`` returns None (Maple's FAIL) instead of
raising, comparisons treat None as false both ways, and an independent-variable
leader is reported as a non-(differential-)reductor (matching factor_deriv's
[1,var] intent). Parameter and jet leaders are unchanged.

Run:  ~/miniforge3/envs/sage/bin/python tests/test_indep_leader.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parametric_rg.ring import build_ring
from parametric_rg.main import PRG
from parametric_rg import diffhelpers as dh

failures = []

ns = build_ring(['x', 'y', 'z'], [['DDPs', 'DPs', 'Ps', 'Vf', 'rho']],
                ['EE', 'V1', 'V2', 'V3', 'V4', 'a0', 'a1', 'b0', 'b1', 'c0', 'c1'])
prg = PRG(ns['R'], ns['derivations'], ns['parms'])
R = prg.R                                   # wrapped ring (production path: _derivations set)
x, y, z = ns['derivations']
rho = ns['jet']('rho')

g = rho**2 - x**2 - y**2 - z**2             # jet-bearing chain element
f = x * y - z                               # coordinate relation, leader = x (independent var)

# deg_in: None (Maple FAIL) for an indep-var gen with a jet present; finite otherwise
if dh.deg_in(R, g, x) is not None:
    failures.append("deg_in(g, x) should be None (Maple FAIL), got %r" % (dh.deg_in(R, g, x),))
if dh.deg_in(R, g, rho) != 2:
    failures.append("deg_in(g, rho) should be 2, got %r" % (dh.deg_in(R, g, rho),))

# isreduceble / leaderreduced: no crash, report not-reducible for the indep-var leader
try:
    if dh.isreduceble(R, g, f) is not False:
        failures.append("isreduceble(g, f) should be False, got %r" % (dh.isreduceble(R, g, f),))
    if dh.leaderreduced(R, g, f) is not False:
        failures.append("leaderreduced(g, f) should be False, got %r" % (dh.leaderreduced(R, g, f),))
except Exception as e:                      # noqa: BLE001
    failures.append("isreduceble/leaderreduced crashed on indep-var leader: %r" % (e,))

# regression: genuine jet leaders still reduce as before
p = ns['jet']('Ps', 'x', 'x') + ns['jet']('Ps', 'x')
q = ns['jet']('Ps', 'x', 'x') - 1
if dh.isreduceble(R, p, q) is not True:
    failures.append("isreduceble(p, q) with genuine jet leaders should be True")

print()
if failures:
    print("FAILURES:")
    for fl in failures:
        print("  -", fl)
    sys.exit(1)
else:
    print("All independent-variable-leader checks passed.")
    sys.exit(0)
