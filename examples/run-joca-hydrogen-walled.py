"""Walled re-run of the hydrogen parametric RG.

Compare against the baseline log (run-joca-hydrogen.py output, preserved on
c200-1 as ~/hydrogen-native-c200.log.run1-baseline). Same ring / lowest-leader
selection / no-base-field / uncapped config as the baseline, but each
per-vertex diffprem runs under a fork wall (wall-clock + child-RSS cap). A
vertex that breaches is recorded to ~/hydrogen-walled-c200.walled and its
branch is skipped, so the tree finishes instead of wedging on the swell
vertex (baseline run 915085 hung 15 h inside one vertex-328 reduction).

Tunable via env: PRG_WALL_S (seconds, default 300), PRG_WALL_RSS_MB (MiB,
default 3000). Log to ~/hydrogen-walled-c200.log; do NOT clobber the baseline.
"""
import os, sys, time, threading, traceback
# Match the baseline run's lowest-leader-first selection (it was launched with
# this env set, not in the script). Without it the worklist traversal — and so
# the whole vertex trajectory — diverges from hydrogen-native-c200.log.run1-baseline.
os.environ.setdefault('PRG_LOWER_LEADER', '1')
# Durably checkpoint each completed cell as it terminates (survives an OOM /
# SIGKILL / interruption); without this the Decom list lives only in memory
# and is printed once at the end. See main.py PRG_DECOM_FILE (commit 185038b).
os.environ.setdefault('PRG_DECOM_FILE', os.path.expanduser('~/hydrogen-walled-c200.decom'))
sys.setrecursionlimit(1_000_000)
sys.path.insert(0, os.path.expanduser('~/parametric-rg'))
import sympy
from parametric_rg.ring import build_ring
from parametric_rg.main import PRG

ns = build_ring(['x', 'y', 'z'], [['DDPs', 'DPs', 'Ps', 'Vf', 'rho']],
                ['EE', 'V1', 'V2', 'V3', 'V4', 'a0', 'a1', 'b0', 'b1', 'c0', 'c1'])
R, der, parms, jet = ns['R'], ns['derivations'], ns['parms'], ns['jet']
x, y, z = der
EE, V1, V2, V3, V4, a0, a1, b0, b1, c0, c1 = parms
DDPs, DPs, Ps, Vf, rho = jet('DDPs'), jet('DPs'), jet('Ps'), jet('Vf'), jet('rho')
ansatz = [jet('Ps', 'x') - DPs * jet('Vf', 'x'), jet('Ps', 'y') - DPs * jet('Vf', 'y'), jet('Ps', 'z') - DPs * jet('Vf', 'z'),
          jet('DPs', 'x') - DDPs * jet('Vf', 'x'), jet('DPs', 'y') - DDPs * jet('Vf', 'y'), jet('DPs', 'z') - DDPs * jet('Vf', 'z'),
          (a0 + a1 * Vf) * DDPs + (b0 + b1 * Vf) * DPs + (c0 + c1 * Vf) * Ps,
          Vf - (V1 * x + V2 * y + V3 * z + V4 * rho), rho**2 - x**2 - y**2 - z**2]

WALL_S = float(os.environ.get('PRG_WALL_S', '300'))
WALL_RSS = int(os.environ.get('PRG_WALL_RSS_MB', '3000'))
WFILE = os.path.expanduser('~/hydrogen-walled-c200.walled')
print("HYDROGEN_WALLED_START wall_s=%g wall_rss_mb=%d  %s"
      % (WALL_S, WALL_RSS, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)

hold = {}
def run():
    s = PRG(R, der, parms); hold['s'] = s; t0 = time.time()
    try:
        de = s.MainProc(ansatz, [], max_vertices=10**9, progress_every=1,
                        diffprem_wall_s=WALL_S, diffprem_wall_rss_mb=WALL_RSS,
                        walled_file=WFILE)
        hold['r'] = (de, time.time() - t0)
    except BaseException as ex:
        hold['e'] = (ex, time.time() - t0, traceback.format_exc())

threading.stack_size(1 << 30); t = threading.Thread(target=run); t.start(); t.join()
s = hold.get('s'); walled = getattr(s, 'walled', []) if s else []
if 'r' in hold:
    de, dt = hold['r']
    print("HYDROGEN_WALLED_DONE %d cells, %d walled  (%.1fs)  %s"
          % (len(de), len(walled), dt, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    for i, (A, S, N, W) in enumerate(de, 1):
        print("cell %d | N=%s | W=%s" % (i, sorted(str(sympy.expand(n)) for n in N),
              sorted(str(sympy.expand(w)) for w in W)), flush=True)
    for w in walled:
        print("WALLED v#%s | %s ld=%s | |A|=%d | in=%d terms | %s | %.1fs"
              % (w['cell_label'], w['kind'], w['leader'], w['nA'], w['in_terms'],
                 w['why'], w['elapsed']), flush=True)
else:
    ex, dt, tb = hold['e']
    print("HYDROGEN_WALLED_STOP %r (%.1fs)" % (ex, dt), flush=True); print(tb, flush=True)
