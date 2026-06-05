# parametric-rg

A Python/Sage port of the **parametric Rosenfeld–Gröbner (RG)** algorithm of

> S. Fakouri, S. Rahmany, A. Basiri, *A new algorithm for computing regular
> representations for radicals of parametric differential ideals*, Cogent
> Mathematics & Statistics **5** (2018) 1507131.
> <https://doi.org/10.1080/25742558.2018.1507131>

It is a faithful translation of Rahmany's Maple implementation
`par-rga.txt` (archived under `~/project/rahmani/`), cross-referenced against
§4 of the paper (Algorithms 4.1–4.8). It runs on the **stock, unmodified**
Python `DifferentialAlgebra` binding (BLAD/BMI) — no library rebuild — by
reconstructing the one primitive the Python binding does not expose,
`DeltaPolynomial`.

## Why

The parametric RG decomposes a parametric differential system `A = 0, S ≠ 0`
over `K[α₁,…,αₜ]{u₁,…,uₙ}` into all distinct **regular representations** for the
different cases of the parameters, each tagged with a Montes quasi-canonical
`(N, W)` pair (zero / non-zero parametric conditions). It branches on the
parameters *explicitly* with cheap primitives instead of running an ordinary
RG with the constants as ring variables (which blows up).

## Layout

```
parametric_rg/
  diffprim.py    reconstructed DeltaPolynomial (+ theta/reductum helpers)
  util.py        GCD, DIVIDE, RZ, SIM, ADD, Extract, PComponent, FACTOR
  paramring.py   the (N,W) side: Sage/Singular QQ[parms] lex Groebner,
                 NormalForm, IdealMembership, RadicalMembership, is_trivial
  diffhelpers.py Test, SC, Sec_Crit (Buchberger criteria), update (Alg 4.6)
  main.py        PRG class: MainProc (4.1) + NewPCondition (4.4),
                 NewCondition (4.5), Branch, MakeTree (4.2), CheckParm (4.7),
                 CheckBranch (4.8), checkregular (Rosenfeld)
  ring.py        ring builder + Maple jet notation (u[x,x] -> Derivative)
  verify.py      input-reduces-to-zero consistency check
examples/
  example1.py    Lorenz P1   (ring F, branches on a)
  example2.py    scalar 3rd-order ODE P2 (one case for all params)
  example3.py    parametric PDE P3 (ring F1; the (a+1)*u_xx^2 case)
  nonparametric_check.py   sanity check vs stock RosenfeldGroebner
  joca-prg.sage  hydrogen ansatz + Schroedinger PDE (NewMethod section 1.12)
tests/
  test_delta.py     DeltaPolynomial vs the BMI Maple oracle (6/6)
  test_examples.py  example1/2/3 case counts + consistency
```

## Running

Everything must go through Sage (the `(N,W)` side needs `sage.all`; the
differential side needs `DifferentialAlgebra`, available in the `sage` env):

```bash
~/miniforge3/envs/sage/bin/sage -python tests/test_delta.py
~/miniforge3/envs/sage/bin/sage -python tests/test_examples.py
~/miniforge3/envs/sage/bin/sage -python examples/example1.py
~/miniforge3/envs/sage/bin/sage examples/joca-prg.sage          # .sage driver
```

The library modules are plain `.py` using Sage-safe syntax (`**` not `^`,
explicit `QQ(...)`/`PolynomialRing(...)`), so `import parametric_rg` works
from inside a `sage` session. Only `examples/*.sage` use the Sage preparser.

## The reconstructed `DeltaPolynomial`

`par-rga.txt` calls `DeltaPolynomial`, exposed by BLAD/BMI and the Maple
binding but **not** the Python binding. `diffprim.DeltaPolynomial` reimplements
it, mirroring the BLAD C routine `bad_delta_polynomial_critical_pair`
(`~/DifferentialAlgebra/blad/bad/src/bad_critical_pair.c`):

For leaders `u₁, u₂` of `p₁, p₂` and their least common derivative `u₁₂`:

* **triangular** (`u₁₂ ≠ u₁, u₂`):
  `Δ = reductum(θ₁·p₁)·s₂ − reductum(θ₂·p₂)·s₁`, where `θᵢ` carries `uᵢ→u₁₂`,
  `sᵢ` is the separant of `pᵢ`, and `s₁,s₂` are first divided by their gcd
  (as BLAD does). This is the classical
  `Δ = s₁·(θ₁₂/θ₂)·p₂ − s₂·(θ₁₂/θ₁)·p₁` restricted below `u₁₂` (the `s·u₁₂`
  terms cancel).
* **non-triangular** (`u₁₂ = u₁` or `u₂`): a Ritt pseudo-remainder of the
  higher-ranked polynomial by the (differentiated) lower one w.r.t. `u₁₂`,
  done with the binding's own `differential_prem`.

It is validated against the BMI Maple test oracle
`~/DifferentialAlgebra/bmi/maple/tests/delta_polynomial.tst` (6/6, up to the
sign freedom in the numeric normalisation). No other Maple `Tools` used by
par-rga are missing from the Python binding.

## Stock-library note

`par-rga`'s `ParmDiffRed` adjoins the parameters' derivatives `αⱼ[xᵢ]` to the
reduction set. On the stock Python binding the parameters are genuine
constants, so their derivatives are literally `0` and contribute nothing; we
drop them. Everything else maps directly onto the binding.

## Status

* `DeltaPolynomial`: 6/6 oracle cases.
* example1 (Lorenz): 4 regular systems = 2 cases (`a=0`, `a≠0`) × 2 components
  each — matching the paper's Table 1 (each case is an intersection of two
  regular ideals).
* example2: exactly 1 system for all parameters — matching paper §5.
* example3 (PDE): 15 systems over cells keyed on `a+1`, `b`, `c`; `a=0`
  correctly excluded (inconsistent) — matching the 8 case-keys of Table 2.
* Hydrogen (joca): the §1.12 cross-check **passes** — all 5 of joca.sage's
  minimal associated primes are exactly the parametric cells on which the PDE
  is redundant modulo the ansatz. The direct parametric-RG run on the ansatz
  is time-boxed by per-vertex coefficient swell (not parameter-tree
  explosion).

See `~/project/reports/parametric-rg-port.md` for the full write-up.
