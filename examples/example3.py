"""Example 3 of Fakouri-Rahmany-Basiri (2018), ported from
~/project/rahmani/example3.txt:

    Mainproc([(a+1)*u[x,x]^2*v + b*u[x,x]*v + c*u[x], u[x,y], a*u[y,y]-1], [], F1)

This is the parametric PDE system P3.  Ring F1 (par-rga.txt):
    derivations [x,y,z], blocks=[[m,u,v,w]] (orderly single block), arbitrary a,b,c,d.

The (a+1) factor is a *parametric initial* of the first equation's leader
u_xx^2, which forces the parameter-branching the algorithm exists to do.
Expected (paper Table 2): EIGHT distinct cases keyed on a, b, c, a+1; the
case a=0 is INCONSISTENT (absent from the output).

Run:  ~/miniforge3/envs/sage/bin/sage -python examples/example3.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parametric_rg.ring import build_ring
from parametric_rg.main import PRG
from parametric_rg.verify import check_systems

ns = build_ring(['x', 'y', 'z'], [['m', 'u', 'v', 'w']], ['a', 'b', 'c', 'd'])
R, derivations, parms, jet = ns['R'], ns['derivations'], ns['parms'], ns['jet']
a, b, c, d = parms
u, vv = ns['dep']['u'], ns['dep']['v']
U = u(*derivations); V = vv(*derivations)

uxx = jet('u', 'x', 'x')
ux = jet('u', 'x')
uxy = jet('u', 'x', 'y')
uyy = jet('u', 'y', 'y')

eqs = [
    (a + 1) * uxx**2 * V + b * uxx * V + c * ux,
    uxy,
    a * uyy - 1,
]

solver = PRG(R, derivations, parms, verbose=True)
t0 = time.time()
result = solver.MainProc(eqs, [], max_vertices=20000)
dt = time.time() - t0

print("=== example3 (PDE P3) result (%.1fs, %d systems) ===" % (dt, len(result)))
for i, sysm in enumerate(result):
    A, S, N, Wn = sysm
    print("--- system %d ---" % i)
    print("  N (params=0)   =", N)
    print("  W (params!=0)  =", Wn)
    print("  A (chain)      =", A)
    print("  S (ineqs)      =", S)

print("\n=== consistency check ===")
check_systems(R, eqs, result, solver.PR)
