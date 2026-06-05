"""Example 2 of Fakouri-Rahmany-Basiri (2018), ported from
~/project/rahmani/example2.txt:

    Mainproc([w-v[x], v-u[x], -w[x]+a*u+b*v+c*w+u^3], F)

Ring (paper section 5): R = Q[a,b,c,d]{m,u,v,w}, orderly ranking w<v<u<m,
elimination ranking m<u<v<w.  Expected output (paper, system P2): ONE case
for all a,b,c:

    sqrt([P2]) = [v - ux, w - uxx, u^3 + au + bv + cuxx - uxxx]

Run:  ~/miniforge3/envs/sage/bin/sage -python examples/example2.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parametric_rg.ring import build_ring
from parametric_rg.main import PRG

# Orderly ranking: single block, highest-ranked variable first (m > u > v > w).
ns = build_ring(['x', 'y', 'z'], [['m', 'u', 'v', 'w']], ['a', 'b', 'c', 'd'])
R, derivations, parms, jet = ns['R'], ns['derivations'], ns['parms'], ns['jet']
a, b, c, d = parms
w, vv, u = ns['dep']['w'], ns['dep']['v'], ns['dep']['u']
W = w(*derivations); V = vv(*derivations); U = u(*derivations)

eqs = [
    W - jet('v', 'x'),
    V - jet('u', 'x'),
    -jet('w', 'x') + a*U + b*V + c*W + U**3,
]

solver = PRG(R, derivations, parms, verbose=True)
t0 = time.time()
result = solver.MainProc(eqs, [])
dt = time.time() - t0

print("=== example2 result (%.1fs, %d systems) ===" % (dt, len(result)))
for i, sysm in enumerate(result):
    A, S, N, Wn = sysm
    print("--- system %d ---" % i)
    print("  A (chain) =", A)
    print("  S (ineqs) =", S)
    print("  N (params=0) =", N)
    print("  W (params!=0) =", Wn)
