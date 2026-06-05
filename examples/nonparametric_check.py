"""Non-parametric sanity check: rga_main.txt system2 vs BLAD RosenfeldGroebner.

system2.txt (rga_main.txt) is the non-parametric analogue of example3:
    RosGrobim([u[x,x]^2*v + u[x,x]*v + u[x], u[x,y], u[y,y]-1], [], R)
with R = DifferentialRing(derivations=[x,y,z], blocks=[[u,v,w,m]]) (orderly).

We run our ported parametric RG (with an empty effective parameter set) and
compare the number / leaders of regular components against the stock binding's
own RosenfeldGroebner.

Run:  ~/miniforge3/envs/sage/bin/sage -python examples/nonparametric_check.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sympy
from sympy import Derivative, var
import DifferentialAlgebra as DA
from parametric_rg.ring import build_ring
from parametric_rg.main import PRG
from parametric_rg.verify import reduces_to_zero

# ---- our port (parameters declared but unused in this system) -------------
ns = build_ring(['x', 'y', 'z'], [['m', 'u', 'v', 'w']], ['a', 'b', 'c', 'd'])
R, derivations, parms, jet = ns['R'], ns['derivations'], ns['parms'], ns['jet']
u, vv = ns['dep']['u'], ns['dep']['v']
V = vv(*derivations)
eqs = [
    jet('u', 'x', 'x')**2 * V + jet('u', 'x', 'x') * V + jet('u', 'x'),
    jet('u', 'x', 'y'),
    jet('u', 'y', 'y') - 1,
]
solver = PRG(R, derivations, parms)
t0 = time.time()
ours = solver.MainProc(eqs, [])
dt = time.time() - t0
print("Our port: %d regular systems (%.2fs)" % (len(ours), dt))
for i, (A, S, N, W) in enumerate(ours):
    print("  comp %d: leaders=%s  N=%s W=%s" %
          (i, [R.leading_derivative(p) for p in A], N, W))
    bad = reduces_to_zero(R, eqs, A, solver.PR, N)
    print("           consistent:", not bad)

# ---- stock BLAD RosenfeldGroebner -----------------------------------------
x, y, z = var('x y z')
uu = sympy.Function('u'); v2 = sympy.Function('v'); w2 = sympy.Function('w')
m2 = sympy.Function('m')
R2 = DA.DifferentialRing(derivations=[x, y, z], blocks=[[u(*derivations).func,
     v2, w2, m2]] if False else [[uu, v2, w2, m2]])
e = [Derivative(uu(x, y, z), x, x)**2 * v2(x, y, z)
     + Derivative(uu(x, y, z), x, x) * v2(x, y, z)
     + Derivative(uu(x, y, z), x),
     Derivative(uu(x, y, z), x, y),
     Derivative(uu(x, y, z), y, y) - 1]
blad = R2.RosenfeldGroebner(e)
print("\nBLAD RosenfeldGroebner: %d regular chains" % len(blad))
for i, chain in enumerate(blad):
    print("  chain %d eqs:" % i, chain.equations())
