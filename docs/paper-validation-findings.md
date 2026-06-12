# Validating the port against Fakouri–Rahmany–Basiri (2018)

**Date:** 2026-06-11
**Scope:** Comparison of this port's output for the paper's three worked
examples (P1 Lorenz, P2 Genesio–Tesi, P3 PDE) against the published results,
plus a faithfulness bug found and partially fixed in the process.

Paper: S. Fakouri, S. Rahmany, A. Basiri, *A new algorithm for computing
regular representations for radicals of parametric differential ideals*,
Cogent Math. & Stat. 5 (2018) 1507131. Local copy:
`~/project/papers/fakouri-2018-parametric-rosenfeld-groebner.pdf`. Maple
original: `~/project/rahmani/par-rga.txt`.

---

## TL;DR

1. **`SIM` was unfaithful** to the Maple original and silently disabled
   `Branch`'s per-factor product split. Fixed in commit `0363799`
   (`SIM: preserve product structure`). The fix is a no-op at every other
   call site (audited) and recovers a real missing split.
2. **The example test suite does not reproduce the paper.** It checks loose
   lower bounds (`>=2`, `>=8`) plus self-consistency, never exact counts or
   per-cell content. example3 produces 16 regular systems where the paper's
   Table 2 has **8 cases / 17 components** — a mismatch the `>=8` floor hides.
3. **`SIM` was necessary but not sufficient** for P3. After the fix, example3
   moves 15 → 16 systems but still diverges from the paper (10 overlapping
   cases vs 8 clean ones; one component still dropped on the generic side).
   A second, distinct divergence remains in the parametric-condition layer.
4. **P2's port output and the paper's printed P2 differ as polynomial sets**
   but are the **same ideal** — a presentation difference, not a bug. Any
   future "matches the paper" assertion must compare *saturated ideals*, not
   literal chains.

---

## 1. The `SIM` faithfulness bug (fixed)

### Maple vs the port

`par-rga.txt:62`:
```maple
SIM:=proc(f)
  if f=0 then return(f): else return( f / GCD([coeffs(expand(f))]) ): fi:
end:
```
Divides by the integer content of `expand(f)` but returns `f` itself divided
by that integer — **preserving f's structure** (a product stays a product).

The port (before the fix) was `expand(primitive(f))` — always re-expands.

### Why it matters: only in `Branch`

`Branch` (`main.py`, `par-rga.txt:271`) factors the polynomial and splits on
each factor:
```python
f = _strip_power(sympy.factor(p))      # e.g. v[x]*(v[z]*DPsi[y] - v[y]*DPsi[z])
if f.func != sympy.Add:
    h = SIM(f)
    if h.func == sympy.Mul:            # Maple: type(h,'*')
        for o in h.args: ...           # one child per non-constant factor
```
With the re-expanding `SIM`, `h` became an `Add`, `h.func == Mul` was always
false, and the per-factor branch was **dead code**. So wherever a differential
reduction produced a *factorable* polynomial, the port made **one** child
where Fakouri's algorithm makes one **per factor** — a coarser decomposition.

### Audit (why fixing `SIM` is safe)

All 18 `SIM` call sites were checked. Every site except `Branch` feeds `SIM`
an already-expanded polynomial (separants, initials, `expand(...)` results,
`diffprem` remainders), for which `f / content` equals `expand(primitive(f))`
exactly — same value, same sign. Only `Branch:245` inspects `SIM`'s output
structure (`h.func`); everywhere else the result flows into
`FACTOR`/`DIVIDE`/`leading_derivative`/`is_constant`/`ADD`, which accept any
expression form. So the fix is a no-op at every currently-exercised site and
changes behaviour only where intended.

### Worked instance (hydrogen, v#6 → v#7)

The Δ-pair `Δ(Psi[x], Psi[z])` reduces (against the existing `DPsi[x]` chain
element) to `v[x]·v[z]·DPsi[y] − v[x]·v[y]·DPsi[z]`, which factors as
`v[x] · (v[z]·DPsi[y] − v[y]·DPsi[z])`. Maple splits this into a `v[x] = 0`
case and the cofactor case (2 children); the pre-fix port produced 1 child,
carrying `v[x]` as a coefficient factor and never exploring `v[x] = 0`.

---

## 2. The example tests do not reproduce the paper

`tests/test_examples.py` asserts, per example, a **count bound** plus a
self-consistency check (`reduces_to_zero`: every returned chain reduces all
inputs to 0 on its cell). The consistency check is one-directional — a
*coarser* decomposition still passes it — so it cannot detect a missing split.

| example | paper result | test assertion | port (post-fix) | matches paper? |
|---|---|---|---|---|
| P1 (Lorenz) | 2 cases × 2 ideals = 4 | `len >= 2` | 4 systems | plausibly (count) |
| P2 (Genesio–Tesi) | 1 case | `len == 1` | 1 system | count ✓; ideal ✓; chain differs (§4) |
| P3 (PDE) | **8 cases / 17 components** | `len >= 8` + `a=0` absent | **16 systems / 10 cases** | **no** |

The `>=8` floor is exactly what let a wrong P3 answer (15 pre-fix, 16
post-fix; should be ~17) pass as green.

---

## 3. P3 after the fix: closer, still wrong

Grouping example3's output by case `(N, W)`:

- **pre-fix:** 9 cases, 15 components.
- **post-fix:** 10 cases, 16 components. The fix split the `c=0` region,
  adding a `c=0, b≠0` cell (`W={a,a+1,b}`, 2 components) — a genuine recovered
  split.

Residual divergences from the paper's Table 2 (8 clean cases / 17 components):

1. **Overlapping `b`-unconstrained cells.** Port case 1 (`W={a,a+1,c}`,
   `b` free) *contains* case 0 (`b≠0`) and case 6 (`b=0`); case 8 (`c=0`,
   `b` free) overlaps case 9. The paper's cases form a partition; the port
   emits redundant coarser cells alongside the finer ones. Suggests the
   parametric-condition layer (`NewPCondition`/`MakeTree`) is not forcing
   `b` into `N` or `W` on the generic branch.
2. **A component still dropped on the generic side.** Case 0: 2 vs paper's 3;
   cases 6, 7: 1 vs paper's 2. The `a+1 = 0` side is a *perfect* match
   (3,2,1,1 components) — the divergence lives entirely on the generic
   `a+1 ≠ 0` side, i.e. where the leading `(a+1)·u_xx²` term is present and
   factorable reductions occur.

Conclusion: `SIM` was *a* cause, not the only one. At least one further
divergence remains, most likely in the parametric branching / `(N,W)`
canonicalization. Next step: instrument `MakeTree`/`NewPCondition` on P3's
generic branch to locate where `b≠0`/`b=0` fails to register and where the
missing component is pruned.

---

## 4. P2: same ideal, different presentation (not a bug)

The paper's P2 representation (one case):
```
[v − u_x,  w − u_xx,  u³ + a·u + b·u_x + c·u_xx − u_xxx]      leaders u_x, u_xx, u_xxx
```
The port returns the **input unchanged**:
```
[w − v_x,  v − u_x,  −w_x + a·u + b·v + c·w + u³]             leaders v_x, u_x, w_x
```
These are **different polynomial sets** but the **same radical differential
ideal** — verified by identical Gröbner bases after prolonging both chains
and treating jets as polynomial variables (`/tmp/prg-trace/ex2_ideal_eq.py`).
The port returns the natural orderly-ranking chain; the paper printed the
`u`-eliminated form (the underlying 3rd-order ODE `u_xxx = a·u + b·u_x +
c·u_xx + u³`) for readability.

Caveat for future tests: `differential_prem`-to-zero is **not** a complete
membership test across chains with different leader sets (it is leader-
directed and can get stuck). Ideal equality must be checked by an algebraic
Gröbner comparison of the *saturated* ideals, not by literal chain equality
and not by one-directional differential reduction.

---

## 5. Recommended test hardening

To make the suite actually validate against the paper:

- Assert **exact** case counts (`== 2`, `== 1`, `== 8`), not lower bounds.
- Assert the per-case `(N, W)` parametric conditions against Table 1 / Table 2.
- Assert per-cell **ideal equality** (Gröbner basis of the saturated ideal)
  against the paper's printed ideals — never literal chain equality (see §4).
- Add a `b`-coverage check on P3: every generic-branch cell must place each
  parameter in `N` or `W` (no unconstrained-`b` overlaps).

---

## Artifacts

- Fix: commit `0363799` (`parametric_rg/util.py`).
- Repro scripts (session-local, `/tmp/prg-trace/`): `ex2.py`,
  `ex2_vs_paper.py`, `ex2_ideal_eq.py`, `ex3_cases.py`, `walk_v6.py`,
  `hydrogen-trace.log` (full `PRG_TRACE` of the hydrogen run).
- Paper: `~/project/papers/fakouri-2018-parametric-rosenfeld-groebner.pdf`.
- Maple original: `~/project/rahmani/par-rga.txt`.

*Researched and written by Claude on behalf of Brent Baccala, 2026-06-11.*
