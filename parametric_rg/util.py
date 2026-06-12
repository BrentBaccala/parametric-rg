"""Maple-helper utilities translated from par-rga.txt.

These mirror the small list/poly helpers GCD, DIVIDE, RZ, SIM, SIM2, ADD,
Extract, PComponent, FACTOR used throughout par-rga.txt.  They operate on
sympy expressions.  ``parms`` (the list of parameter symbols) is passed
explicitly where the Maple original closed over a global ``parms``.
"""

import sympy
from sympy import expand, gcd, factor_list, Integer, S, Poly, cancel


def GCD(L):
    """gcd of the members of list L (Maple GCD)."""
    if not L:
        return S.Zero
    g = sympy.sympify(L[0])
    for x in L[1:]:
        g = gcd(g, sympy.sympify(x))
    return g


def DIVIDE(f, U):
    """Divide f by u^m for each u in U while u | f  (Maple DIVIDE).

    Repeatedly strips any factor that f shares with a member of U."""
    f = sympy.sympify(f)
    if f == 0:
        return S.Zero
    q = f
    for u in U:
        u = sympy.sympify(u)
        if u == 0:
            continue
        sw = True
        while sw:
            g = gcd(q, u)
            if g != 1 and g != 0:
                q = expand(cancel(q / g))
            else:
                sw = False
    return q


def RZ(L):
    """Remove zeros from list L."""
    return [x for x in L if sympy.sympify(x) != 0]


def SIM(f):
    """Maple SIM: divide by the integer content of ``expand(f)`` but PRESERVE
    f's structure -- a factored product stays a product.

    Faithful to par-rga's ``f / GCD([coeffs(expand(f))])`` (NOT the earlier
    ``expand(primitive(f))``).  The two agree on any already-expanded input;
    they differ only when ``f`` is a product, which happens deliberately in
    ``Branch`` where SIM is applied to ``factor(p)`` and the product structure
    (``type(h,'*')`` / ``h.func == Mul``) decides the per-factor branching.
    Re-expanding there silently killed that branch."""
    f = sympy.sympify(f)
    if f == 0:
        return S.Zero
    try:
        cont, _prim = sympy.primitive(expand(f))
    except Exception:
        return f
    if cont == 0 or cont == 1:
        return f
    return f / cont


def ADD(L, M):
    """Append members of M to L avoiding repeats (up to sign).  (Maple ADD)."""
    L = list(L)
    s = list(L)
    for m in M:
        m = sympy.sympify(m)
        if not any(_same(m, x) or _same(-m, x) for x in s):
            s.append(m)
    return s


def _same(a, b):
    # structural comparison for critical pairs (tuples/lists of polynomials)
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        if not (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list))):
            return False
        if len(a) != len(b):
            return False
        return all(_same(x, y) for x, y in zip(a, b))
    return expand(sympy.sympify(a) - sympy.sympify(b)) == 0


def Extract(L, T):
    """Return members of L not equal to any member of T  (Maple Extract)."""
    if not T:
        return list(L)
    out = []
    for x in L:
        if not any(_same(x, t) for t in T):
            out.append(x)
    return out


def member(x, L):
    return any(_same(x, y) for y in L)


def PComponent(p):
    """Prime (irreducible, multiplicity-free) components of p  (Maple PComponent)."""
    p = sympy.sympify(p)
    coeff, facs = factor_list(p)
    h = []
    for base, _exp in facs:
        if not base.is_constant():
            h.append(expand(base))
    if not h:
        return [expand(p)]
    return h


def FACTOR(f, parmset):
    """Product of the irreducible factors of f involving only parameters.

    Returns False if there is no such (non-constant) factor.  (Maple FACTOR)."""
    f = sympy.sympify(f)
    v = f.free_symbols
    if v <= set(parmset):
        return f
    coeff, facs = factor_list(f)
    kg = Integer(1)
    found = False
    for base, exp in facs:
        if base.free_symbols <= set(parmset) and not base.is_constant():
            kg = kg * base**exp
            found = True
    if not found or sympy.sympify(kg).is_constant():
        return False
    return expand(kg)


def is_constant(f):
    f = sympy.sympify(f)
    return f.is_number
