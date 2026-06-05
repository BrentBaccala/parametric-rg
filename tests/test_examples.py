"""Regression test for Rahmany's example1/2/3 via the ported parametric RG.

Asserts each runs to completion, the case/cell structure matches the paper,
and every returned system reduces all inputs to 0 on its cell.

Run:  ~/miniforge3/envs/sage/bin/sage -python tests/test_examples.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parametric_rg.ring import build_ring
from parametric_rg.main import PRG
from parametric_rg.verify import reduces_to_zero

failures = []


def run_case(name, ring_blocks, build_eqs, expect_min_systems,
             check_a0_absent=False):
    ns = build_ring(['x', 'y', 'z'], ring_blocks, ['a', 'b', 'c', 'd'])
    R, derivations, parms, jet = ns['R'], ns['derivations'], ns['parms'], ns['jet']
    eqs = build_eqs(ns)
    solver = PRG(R, derivations, parms)
    result = solver.MainProc(eqs, [])
    print("[%s] %d systems" % (name, len(result)))
    # consistency: each system reduces all inputs to 0 on its cell
    for idx, (A, S, N, W) in enumerate(result):
        bad = reduces_to_zero(R, eqs, A, solver.PR, N)
        if bad:
            failures.append("%s sys%d not reduced: %s" % (name, idx, bad))
    if len(result) < expect_min_systems:
        failures.append("%s: expected >= %d systems, got %d"
                        % (name, expect_min_systems, len(result)))
    if check_a0_absent:
        import sympy
        a = parms[0]
        for (A, S, N, W) in result:
            # no cell should force a = 0 (a in N) without it being inconsistent
            if any(sympy.sympify(n) == a for n in N):
                failures.append("%s: a=0 cell present (should be inconsistent)" % name)
    return result


def ex1(ns):
    a, b, c, d = ns['parms']
    w, vv, u = ns['dep']['w'], ns['dep']['v'], ns['dep']['u']
    der = ns['derivations']; jet = ns['jet']
    W, V, U = w(*der), vv(*der), u(*der)
    return [jet('u', 'x') - a*(V - U),
            jet('v', 'x') - U*(b - W) + V,
            jet('w', 'x') - U*V + c*W]


def ex2(ns):
    a, b, c, d = ns['parms']
    w, vv, u = ns['dep']['w'], ns['dep']['v'], ns['dep']['u']
    der = ns['derivations']; jet = ns['jet']
    W, V, U = w(*der), vv(*der), u(*der)
    return [W - jet('v', 'x'), V - jet('u', 'x'),
            -jet('w', 'x') + a*U + b*V + c*W + U**3]


def ex3(ns):
    a, b, c, d = ns['parms']
    u, vv = ns['dep']['u'], ns['dep']['v']
    der = ns['derivations']; jet = ns['jet']
    U, V = u(*der), vv(*der)
    return [(a+1)*jet('u','x','x')**2*V + b*jet('u','x','x')*V + c*jet('u','x'),
            jet('u', 'x', 'y'),
            a*jet('u', 'y', 'y') - 1]


# example1: ring F (blocks w,v,u,m), >=2 cases (a!=0, a=0), each an intersection
run_case("example1", [['w'], ['v'], ['u'], ['m']], ex1, expect_min_systems=2)
# example2: ring F, exactly 1 system for all params
r2 = run_case("example2", [['m', 'u', 'v', 'w']], ex2, expect_min_systems=1)
if len(r2) != 1:
    failures.append("example2: expected exactly 1 system, got %d" % len(r2))
# example3: ring F1 (single orderly block), >=8 cases, a=0 absent
run_case("example3", [['m', 'u', 'v', 'w']], ex3, expect_min_systems=8,
         check_a0_absent=True)

print()
if failures:
    print("FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
else:
    print("All example checks passed.")
    sys.exit(0)
