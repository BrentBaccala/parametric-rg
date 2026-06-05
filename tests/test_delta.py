"""Validate the reconstructed DeltaPolynomial against the BMI Maple oracle.

Oracle: ~/DifferentialAlgebra/bmi/maple/tests/delta_polynomial.tst
Ring there: DifferentialRing(derivations=[x,y], blocks=[u,v]).

Run with:  ~/miniforge3/envs/sage/bin/sage -python tests/test_delta.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sympy
from sympy import Derivative, var, expand
import DifferentialAlgebra as DA
from parametric_rg.diffprim import DeltaPolynomial, normalize_numeric

x, y = var('x y')
u = sympy.Function('u')
v = sympy.Function('v')
R = DA.DifferentialRing(derivations=[x, y], blocks=[u, v])
derivations = [x, y]

ux  = Derivative(u(x, y), x)
uy  = Derivative(u(x, y), y)
uxx = Derivative(u(x, y), x, x)
uxy = Derivative(u(x, y), x, y)


def delta(p, q):
    return DeltaPolynomial(R, p, q, derivations)


def norm(e):
    return normalize_numeric(expand(sympy.sympify(e)))


cases = [
    # (description, p, q, expected)  -- expected from delta_polynomial.tst
    ("20", ux, uy, sympy.Integer(0)),
    ("30", ux, uy - x, sympy.Integer(1)),
    ("60", ux**2 - 4*u(x, y), uy**3 - uy + u(x, y),
     ux**2 + 6*uy**3 - 2*uy),
    ("70", uy**3 - uy + u(x, y), ux**2 - 4*u(x, y),
     ux**2 + 6*uy**3 - 2*uy),
    ("80", ux**2 - 4*u(x, y), uxx, ux),
    ("90", uxx, ux**2 - 4*u(x, y), ux),
]

passed = 0
failed = 0
for name, p, q, expected in cases:
    got = delta(p, q)
    exp_n = norm(expected)
    # Delta is defined up to numeric sign; compare up to +/- after normalisation
    ok = (expand(got - exp_n) == 0) or (expand(got + exp_n) == 0)
    status = "OK " if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print("[%s] case %s:  Delta(p,q) = %s   expected %s" %
          (status, name, got, exp_n))

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
