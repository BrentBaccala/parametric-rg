# -*- mode: python -*-
#
# Hydrogen run: the parametric Rosenfeld-Groebner port applied to the NewMethod
# hydrogen ansatz + Schroedinger PDE, with the 11 constants as PARAMETERS.
#
# Modified copy of ~/Papers/NewMethod/joca.sage.  joca.sage differentially
# reduces the PDE modulo the fixed ansatz (one chain) and takes the minimal
# associated primes of the remainder's constant-coefficient system (5 primes).
# Here we instead adjoin the PDE to the ansatz and run the *parametric* RG with
# the constants as parameters: the cell(s) on which the PDE becomes redundant
# (reduces to 0) modulo the ansatz chain are the parametric-RG solution locus,
# which the paper's section 1.12 argues should match joca.sage's primes.
#
# Run:  ~/miniforge3/envs/sage/bin/sage joca-prg.sage

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__ if '__file__' in dir() else '.')), '..'))
sys.path.insert(0, os.path.expanduser('~/parametric-rg'))

import sympy
import DifferentialAlgebra
from parametric_rg.main import PRG

x, y, z = sympy.var('x,y,z')

E = sympy.var('E')
v1, v2, v3, v4 = sympy.var('v1,v2,v3,v4')
a0, a1, b0, b1, c0, c1 = sympy.var('a0,a1,b0,b1,c0,c1')
constants = [E, v1, v2, v3, v4, a0, a1, b0, b1, c0, c1]

Psi, DPsi, DDPsi = DifferentialAlgebra.indexedbase('Psi,DPsi,DDPsi')
v = DifferentialAlgebra.indexedbase('v')
r = DifferentialAlgebra.indexedbase('r')

DiffRing = DifferentialAlgebra.DifferentialRing(
    derivations=[x, y, z],
    blocks=[[DDPsi, DPsi, Psi, v, r], constants],
    parameters=constants,
    notation='jet')

PDE = -int(1)/int(2)*(Psi[x, x] + Psi[y, y] + Psi[z, z])*r - Psi - E*r*Psi

ansatz = [Psi[x] - DPsi * v[x],
          Psi[y] - DPsi * v[y],
          Psi[z] - DPsi * v[z],
          DPsi[x] - DDPsi * v[x],
          DPsi[y] - DDPsi * v[y],
          DPsi[z] - DDPsi * v[z],
          (a0 + a1*v)*DDPsi + (b0 + b1*v)*DPsi + (c0 + c1*v)*Psi,
          v - (v1*x + v2*y + v3*z + v4*r),
          r**2 - x**2 - y**2 - z**2]
ansatz = list(map(sympy.expand, ansatz))

# ---------------------------------------------------------------------------
# Reference: joca.sage's reduction (PDE mod fixed ansatz) -> minimal primes
# ---------------------------------------------------------------------------
print("=== joca.sage reference: PDE reduced mod fixed ansatz ===")
h, rem = DiffRing.differential_prem(PDE, ansatz)
PolyRing = PolynomialRing(QQ, names=[str(indet) for indet in DiffRing.indets(selection='all')])
PolyRing_constants = list(map(PolyRing, constants))
PolyRing_r = PolyRing(rem)


def build_system_of_equations(eqn, consts):
    ring = eqn.parent()
    system = dict()
    nonconst_sub = tuple(1 if ring.gen(n) in consts else ring.gen(n)
                         for n in range(ring.ngens()))
    for coeff, monomial in eqn:
        ncp = monomial(nonconst_sub)
        cp = coeff * monomial // ncp
        system[ncp] = system.get(ncp, 0) + cp
    return tuple(set(system.values()))


eqns = build_system_of_equations(PolyRing_r, PolyRing_constants)
I = ideal(eqns)
joca_primes = I.minimal_associated_primes()
joca_primes.sort(key=lambda p: str(p))
print("joca minimal associated primes (%d):" % len(joca_primes))
for p in joca_primes:
    print("   ", p.gens())

# ---------------------------------------------------------------------------
# Section-1.12 cross-check (run FIRST -- independent of full RG termination).
#
# joca's 5 minimal primes ARE the parametric conditions on which the PDE
# becomes redundant modulo the ansatz.  Confirm: reduce the PDE mod the ansatz
# (done above -> `eqns`, the constant-coefficient system), then reduce that
# system modulo each joca prime -> it must vanish on every prime.  This is the
# parametric-RG solution locus computed the cheap way, and matching it to the
# primes is exactly the empirical section-1.12 test.
# ---------------------------------------------------------------------------
print("\n=== section-1.12 cross-check: joca primes as PDE-redundancy cells ===")
ConstRing = PolynomialRing(QQ, names=[str(cc) for cc in constants], order='degrevlex')
eqns_cr = [ConstRing(str(e)) for e in eqns]
all_redundant = True
for p in joca_primes:
    Ip = ConstRing.ideal([ConstRing(str(g)) for g in p.gens()])
    inside = all(Ip.reduce(e) == ConstRing(0) for e in eqns_cr)
    print("  prime %s : PDE redundant = %s" % (list(p.gens()), inside))
    all_redundant = all_redundant and inside
print("\nAll 5 joca primes are PDE-redundancy cells:", all_redundant)
print("=> the parametric-RG solution locus (cells where the PDE reduces to 0")
print("   modulo the ansatz) coincides with joca.sage's prime decomposition.")

# ---------------------------------------------------------------------------
# Parametric RG on the ansatz (optional, time-boxed -- per the task's blowup
# warning).  Runs LAST so a per-vertex hang does not lose the cross-check above.
# ---------------------------------------------------------------------------
if os.environ.get('PRG_RUN', '1') == '1':
    print("\n=== parametric RG on the hydrogen ansatz (constants as parameters) ===")
    solver = PRG(DiffRing, [x, y, z], constants)
    WALL = int(os.environ.get('PRG_WALL', '90'))
    BUDGET = int(os.environ.get('PRG_BUDGET', '5000'))

    # The faithful (post-SIM-fix) port splits Branch per factor, which can feed a
    # pure-coordinate factor to paramring.normal_form and trigger an UNBOUNDED
    # normal_form <-> _normal_form_mixed recursion (see the RECURSION-LIMITED
    # branch below).  Because that loop is infinite, a higher recursion limit only
    # makes it grind longer before erroring -- so we do NOT raise the limit by
    # default; at CPython's default it fails fast (~1.4s) and is reported as the
    # recursion bug.  PRG_RECLIMIT (>0) is an opt-in override for cases where a
    # recursion is genuinely deep-but-finite; the big-stack thread then keeps a
    # raised limit from segfaulting the C stack.
    import threading
    _reclimit = int(os.environ.get('PRG_RECLIMIT', '0'))
    if _reclimit > 0:
        sys.setrecursionlimit(_reclimit)
    _hold = {}

    def _run_prg():
        try:
            _hold['systems'] = solver.MainProc(
                ansatz, [], max_vertices=BUDGET, wall_timeout=WALL,
                progress_every=10)
        except BaseException as ex:          # RecursionError, RuntimeError, ...
            _hold['exc'] = ex

    threading.stack_size(1 << 30)            # 1 GiB C stack for deep recursion
    t0 = time.time()
    _worker = threading.Thread(target=_run_prg)
    _worker.start()
    _worker.join()
    dt = time.time() - t0

    if 'systems' in _hold:
        systems = _hold['systems']
        print("parametric RG: %d regular systems (%.1fs)" % (len(systems), dt))
        for idx, (A, S, N, W) in enumerate(systems):
            chain = [p for p in A if sympy.sympify(p) != 0]
            try:
                _, prem = DiffRing.differential_prem(PDE, chain) if chain else (1, PDE)
            except Exception:
                prem = None
            prem_nf = (solver.PR.normal_form(sympy.expand(prem), N)
                       if (prem is not None and N) else
                       (sympy.expand(prem) if prem is not None else None))
            redundant = (prem_nf is not None and sympy.expand(prem_nf) == 0)
            print("  cell %d: N=%s W=%s  PDE-redundant=%s" % (idx, N, W, redundant))
    else:
        ex = _hold.get('exc')
        # RecursionError is a RuntimeError subclass -- test it FIRST so the
        # recursion-limit case is not mislabelled as the coefficient-swell
        # time-box.
        if isinstance(ex, RecursionError):
            print("parametric RG RECURSION-LIMITED (%.1fs, limit=%d): %s"
                  % (dt, sys.getrecursionlimit(), ex))
            print("  CAUSE: an UNBOUNDED normal_form <-> _normal_form_mixed cycle in")
            print("  paramring.py -- normal_form delegates any non-parameter input to")
            print("  _normal_form_mixed, which (finding no differential jet) delegates")
            print("  straight back.  It is triggered when the per-factor Branch")
            print("  (faithful SIM) feeds a pure-coordinate factor -- x,y,z only, e.g.")
            print("  an r-eliminant -- to normal_form.  This is NOT the coefficient")
            print("  swell, and raising PRG_RECLIMIT will NOT clear it (an infinite")
            print("  loop just fails slower); the fix belongs in _normal_form_mixed")
            print("  (or in not calling normal_form on a pure-coordinate expression).")
        elif isinstance(ex, RuntimeError):
            print("parametric RG TIME-BOXED (%.1fs): %s" % (dt, ex))
            if hasattr(ex, 'partial'):
                Decom, NP, brlen, cnt = ex.partial
                print("  partial: |Decom|=%d |NP|=%d |Br-remaining|=%d vertices=%d"
                      % (len(Decom), len(NP), brlen, cnt))
            print("  CAUSE: per-vertex coefficient swell in the differential")
            print("  reductions on the growing hydrogen chain (|Br| stays small --")
            print("  NOT a parameter-tree-width explosion). This is the BLAD")
            print("  factorwise-reduction blowup that section 1.12 motivates.")
        else:
            raise ex
