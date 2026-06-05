"""Example 1 of Fakouri-Rahmany-Basiri (2018), ported from
~/project/rahmani/example1.txt:

    Mainproc([u[x]-a*(v-u), v[x]-u*(b-w)+v, w[x]-u*v+c*w], F)

This is the Lorenz system P1.  Ring (paper section 5): orderly ranking
w<v<u<m over Q[a,b,c,d].  Expected (paper Table 1): TWO cases:
  * a != 0
  * a = 0

Run:  ~/miniforge3/envs/sage/bin/sage -python examples/example1.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parametric_rg.ring import build_ring
from parametric_rg.main import PRG
from parametric_rg.verify import check_systems

# Ring F of par-rga.txt: derivations [x,y,z], blocks=[w,v,u,m] (separate
# blocks, w highest -- the elimination-style ranking the Maple driver uses).
# With this ranking eq1's leader is v with parametric initial -a, which drives
# the a!=0 / a=0 branching (paper Table 1).
ns = build_ring(['x', 'y', 'z'], [['w'], ['v'], ['u'], ['m']], ['a', 'b', 'c', 'd'])
R, derivations, parms, jet = ns['R'], ns['derivations'], ns['parms'], ns['jet']
a, b, c, d = parms
w, vv, u = ns['dep']['w'], ns['dep']['v'], ns['dep']['u']
W = w(*derivations); V = vv(*derivations); U = u(*derivations)

eqs = [
    jet('u', 'x') - a*(V - U),
    jet('v', 'x') - U*(b - W) + V,
    jet('w', 'x') - U*V + c*W,
]

solver = PRG(R, derivations, parms, verbose=True)
t0 = time.time()
result = solver.MainProc(eqs)
dt = time.time() - t0

print("=== example1 (Lorenz P1) result (%.1fs, %d systems) ===" % (dt, len(result)))
for i, sysm in enumerate(result):
    A, S, N, Wn = sysm
    print("--- system %d ---" % i)
    print("  N (params=0)   =", N)
    print("  W (params!=0)  =", Wn)
    print("  A (chain)      =", A)
    print("  S (ineqs)      =", S)

print("\n=== consistency check ===")
check_systems(R, eqs, result, solver.PR)
