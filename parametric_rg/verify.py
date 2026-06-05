"""Verification helpers for parametric-RG output.

A returned representation [A, S, N, W] is a regular differential system on the
parameter cell { N = 0, W != 0 }.  We spot-check that the *input* equations
reduce to 0 modulo the chain A on that cell (consistency / coverage), as the
task requires.
"""

import sympy


def reduces_to_zero(R, eqs, chain, paramring=None, N=None):
    """Return list of (eq_index, remainder) for inputs NOT reduced to 0 by chain.

    If ``paramring`` and ``N`` are given, the remainder is additionally reduced
    modulo the zero-parameter conditions N before the zero-test (so an equation
    that vanishes only on the cell N=0 is still counted as reduced)."""
    bad = []
    chain = [c for c in chain if sympy.sympify(c) != 0]
    for i, e in enumerate(eqs):
        if not chain:
            r = sympy.expand(sympy.sympify(e))
        else:
            h, r = R.differential_prem(e, chain)
        r = sympy.expand(r)
        if r != 0 and paramring is not None and N:
            r = paramring.normal_form(r, N)
        if sympy.sympify(r) != 0:
            bad.append((i, r))
    return bad


def check_systems(R, eqs, systems, paramring=None):
    """Print/collect a per-system consistency report.  Returns True if every
    system reduces all inputs to 0 on its cell."""
    ok = True
    for idx, (A, S, N, W) in enumerate(systems):
        bad = reduces_to_zero(R, eqs, A, paramring, N)
        status = "OK" if not bad else "FAIL"
        print("  system %d (N=%s, W=%s): %s" % (idx, N, W, status))
        if bad:
            ok = False
            for i, r in bad:
                print("    input %d -> %s" % (i, r))
    return ok
