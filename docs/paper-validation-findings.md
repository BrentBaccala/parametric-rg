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
   call site (audited) and is faithful to Maple — but on P3 its only effect
   was to add one more *redundant* component (15→16), not to approach the
   paper (see §3).
2. **The example test suite does not reproduce the paper.** It checks loose
   lower bounds (`>=2`, `>=8`) plus self-consistency, never exact counts or
   per-cell content. example3 produces 16 regular systems where the paper's
   Table 2 has **8 cases / 17 listings** — a mismatch the `>=8` floor hides.
3. **P3 root cause: the port OVER-produces, it does not under-produce.** vs
   stock `RosenfeldGroebner` the port has *extra* components — redundant `v=0`
   branches (initial-degenerate case of the non-monic first equation),
   contained in broader ones. Fakouri's algorithm has **no** redundancy
   removal (`par-rga.txt` confirmed: only `checkregular`, which drops
   *inconsistent* not *contained* systems); the paper's tables are a minimal /
   curated presentation. An opt-in `remove_redundant` post-pass (§3.1) is
   sound but over-collapses, so it doesn't reproduce the paper either.
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

## 3. P3 root cause: the port OVER-produces (redundant components)

Earlier drafts of this doc guessed the port *under*-produced (a "dropped
component", "factorable reductions"). **That was wrong** — ground truth
(stock `RosenfeldGroebner` at specialised points) reverses it:

| point | stock RG | port (cells valid) | extra |
|---|---|---|---|
| `(2,3,5)` generic | 3 | **4** | `[v, ux, uyy]` (v=0) |
| `(2,0,5)` b=0 | 2 | **3** | `[v, ux, uyy]` |
| `(2,3,0)` c=0 | 3 | 3 | — |
| `(-1,3,5)` a+1=0 | 2 | **3** | `[v, ux, uyy]` |

The port's decomposition is **complete and correct** (same solution set as
stock RG) but **non-minimal**: it carries redundant `v=0` components, each
strictly contained in a broader one (`{v=0,ux=0,auyy=1} ⊊ {ux=0,auyy=1}`,
proved by the regular-chain membership test — *not* the Ritt problem).

**Where the `v=0` branch comes from (traced):** the first equation
`(a+1)·u_xx²·v + b·u_xx·v + c·u_x` is degree 2 in its leader `u_xx`, with
initial `(a+1)·v`. At the root, `Branch`'s two-conditions arm spawns the
**initial-degenerate** branch `initial = 0`, i.e. `(a+1)v = 0` → (on
`a+1≠0`) `v = 0`. It's a *legitimate* branch (a non-monic leader's
coefficient may vanish), created at the root where `Sineq` is **empty** — so
Proposition 4.2 has no inequation to forbid it. It then yields the redundant
`{v=0,ux=0,auyy=1}`.

**Why the paper's Table 2 omits it — there is no algorithmic pruning.**
`grep` confirms `par-rga.txt` has **no** redundancy / inclusion / containment
/ minimality step; `MainProc`'s only filter is `checkregular`, which removes
**inconsistent** systems (`1 ∈ [A]:S^∞`), never **redundant** (contained)
ones. The `v=0` component is consistent, so it survives. Proposition 4.2 (the
paper's "criterion for reducing ineffectual branches") removes a *different*
class — factors forbidden by an inequation already in `S` — and does not
touch these. So **the paper's tables are a minimal / curated presentation**;
the authors evidently dropped the contained components (cross-checked against
BLAD, which *does* eliminate redundancy → 3 not 4). There is no algorithmic
justification for the removal inside the published method.

Corollary: the `SIM` fix (§1) is faithful to Maple but its only P3 effect was
to add *one more* redundant component (15→16); it did not move toward the
paper, because the paper's gap is redundancy, which `SIM` does not govern.

### 3.1 The `remove_redundant` post-pass (opt-in, experimental)

`MainProc(..., minimal=True)` (default **off**) drops a component `C` when the
others that differentially contain it (`V(C) ⊆ V(A_i)`) cover `cell(C)` in
parameter space (a constructible cell-cover check over `ParamRing`). It is the
principled, *decidable* containment removal — but it **does not reproduce the
paper's tables**, for two reasons established by testing:

1. **It over-collapses.** Where a characterizable component is reducible
   (`V = V(P₁) ∪ V(P₂)`), it keeps only the coarse cover and drops the finer
   prime-ish pieces the paper lists separately. Verified **sound** on P1/P3
   (the kept union still covers stock RG's components at specialised points),
   but the result is *coarser* than the paper: P1 `4→2`, P3 `16→13`.
2. **Parametric soundness is uncertified.** `_diff_contains` reduces over
   `ℚ(params)`, so it could in principle report a containment that holds only
   generically, not on all of `cell(C) ∩ cell(A_i)`.

So matching the paper's exact granularity is *between* the raw output
(over-produces, redundant) and this post-pass (over-collapses, coarser): it
needs cell-refined, prime-level irredundancy, not global `V`-containment.
Left as opt-in/experimental; the default stays faithful to `par-rga.txt`.

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
