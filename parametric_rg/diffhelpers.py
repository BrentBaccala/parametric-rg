"""Differential helper sub-routines ported from par-rga.txt.

These wrap the Python DifferentialAlgebra binding and the util helpers to
reproduce par-rga's reduction predicates (isreduceble, leaderreduced),
the Buchberger-criteria filters (Test, SC, Sec_Crit), and the ``update``
sub-algorithm (the analogue of paper Algorithm 4.6).
"""

import sympy
from sympy import Poly, expand, gcd, S, Integer
from .util import GCD, SIM, ADD, Extract, member, _same


def deg_in(R, p, leader):
    """Degree of p in ``leader``, mirroring Maple's ``degree(p, leader)``.

    Returns None (Maple's ``FAIL``) when p is not a polynomial in ``leader`` --
    i.e. when ``leader`` is an independent variable occurring inside a jet's
    arguments, e.g. degree(rho(x,y,z)**2 - x**2, x) = FAIL in Maple. sympy's
    Poly *raises* PolynomialError there, so we catch it and return None; callers
    must treat None like Maple treats FAIL (every ``<=``/``>`` against it is
    false). par-rga relies on exactly this: Maple takes the false branch on
    ``n <= FAIL`` and falls through to the theta-factor test."""
    try:
        return Poly(expand(sympy.sympify(p)), leader).degree()
    except sympy.PolynomialError:
        return None


def factor_deriv(R, deriv):
    """[theta, symb] as a 2-list (Maple FactorDerivative).

    An independent variable (a derivation) is not a dependent-variable jet; the
    binding raises 'dependent variable expected' on it.  Treat it as its own
    trivial factor [1, deriv] so leaderreduced / isreduceble / update see it as
    a non-jet (hence non-reducible, forming no differential critical pairs)."""
    try:
        theta, symb = R.factor_derivative(deriv)
    except RuntimeError:
        return [Integer(1), deriv]
    return [sympy.sympify(theta), symb]


def leader(R, p):
    return R.leading_derivative(p)


def leaderreduced(R, p, q):
    """True if leader(p) is reducible w.r.t leader(q)  (Maple leaderreduced)."""
    dp = R.leading_derivative(p)
    dq = R.leading_derivative(q)
    if dp in _indep(R) or dq in _indep(R):
        # an independent-variable (coordinate-relation) leader is not a
        # differential reductor; see the note in isreduceble. Parameter leaders
        # are not guarded (Poly is well-defined for them).
        return False
    fdp = factor_deriv(R, dp)
    fdq = factor_deriv(R, dq)
    qd, pd = deg_in(R, q, dq), deg_in(R, p, dp)
    cmp_ok = qd is not None and pd is not None   # Maple FAIL -> both <= and > false
    if dp == dq and cmp_ok and qd <= pd:
        return True
    elif dp == dq and cmp_ok and qd > pd:
        return False
    elif fdp[1] == fdq[1] and _divides(fdq[0], fdp[0]):
        return True
    return False


def isreduceble(R, p, q):
    """True if p is reducible w.r.t q (any derivative of p)  (Maple isreduceble)."""
    H = list(sympy.sympify(p).atoms(sympy.Derivative)) + \
        [s for s in sympy.sympify(p).free_symbols]
    dq = R.leading_derivative(q)
    if dq in _indep(R):
        # q's leader is an INDEPENDENT VARIABLE: q is a coordinate relation, not
        # a differential reductor. This is the only leader kind that crashes the
        # port (deg_in -> Poly(p, x) raises when a jet of x is present) and the
        # original (FactorDerivative(x) raises 'dependent variable expected',
        # verified in Maple). factor_deriv's [1,var] convention is meant to make
        # such leaders non-reducible, so report not-reducible rather than
        # spuriously matching the theta branch ([1,x]==[1,x]). NOTE: parameter
        # leaders (also bare symbols) are deliberately NOT guarded -- Poly(p,
        # param) is well-defined, so they keep their original behaviour.
        return False
    fdq = factor_deriv(R, dq)
    degq = deg_in(R, q, dq)
    for dp in H:
        # only consider genuine derivatives / dependent variables
        try:
            fdp = factor_deriv(R, dp)
        except Exception:
            continue
        degp = deg_in(R, p, dp)
        cmp_ok = degq is not None and degp is not None   # Maple FAIL -> both <= and > false
        if dp == dq and cmp_ok and degq <= degp:
            return True
        elif dp == dq and cmp_ok and degq > degp:
            continue
        elif fdp[1] == fdq[1] and _divides(fdq[0], fdp[0]):
            return True
    return False


def _divides(a, b):
    """True if monomial a divides monomial b (both monomials in indep vars)."""
    a = sympy.sympify(a)
    b = sympy.sympify(b)
    if a == 0:
        return False
    q = sympy.simplify(b / a)
    return q.is_polynomial() and sympy.together(b / a).is_polynomial() and \
        (sympy.fraction(sympy.cancel(b / a))[1] == 1)


# ---------------------------------------------------------------------------
# Buchberger criteria
# ---------------------------------------------------------------------------

def Test(R, f, parms):
    """Maple Test: first-criterion eligibility of f.

    f must involve a single dependent variable (one indeterminate besides
    parameters), be non-constant term-wise, have constant initials, and all
    its terms' leaders share the same dependent variable with equal theta
    degree.  Simplified faithful port."""
    f = expand(sympy.sympify(f))
    indt = f.free_symbols | f.atoms(sympy.Derivative)
    indt = {s for s in indt if s not in set(parms)}
    if len(indt) != 1:
        return False
    if f.func == sympy.Add:
        H = list(f.args)
    else:
        H = [f]
    if not all(not sympy.sympify(h).is_constant() for h in H):
        # {type constant} <> {false}  =>  some term is constant => return false
        if any(sympy.sympify(h).is_constant() for h in H):
            return False
    if not all(sympy.sympify(R.initial(h)).is_constant() for h in H):
        return False
    degs = set()
    for h in H:
        ld = R.leading_derivative(SIM(h))
        theta = factor_deriv(R, ld)[0]
        degs.add(sympy.total_degree(sympy.Poly(theta, *_indep(R)))
                 if theta != 1 else 0)
    if len(degs) != 1:
        return False
    return True


def _indep(R):
    # independent variables of the ring; recover from indets that are Symbols
    # used as derivations. We store them on the ring wrapper instead.
    return getattr(R, '_derivations', [])


def SC(R, P, parms):
    """Maple SC: second Buchberger criterion on a triple P=[p1,p2,p3]."""
    H = [factor_deriv(R, R.leading_derivative(SIM(P[i])))[0] for i in range(3)]
    if len(set(map(sympy.srepr, H))) != 3:
        return False
    if GCD(H) == 1:
        return False
    if not _divides(H[1], sympy.lcm(H[0], H[2])):
        return False
    # if no H[i] divides H[j] (i!=j) then True
    any_div = False
    for i in range(3):
        for j in range(3):
            if i != j and _divides(H[j], H[i]):
                any_div = True
    if not any_div:
        return True
    S = R.sort(P, 'ascending')   # binding accepts 'ascending'/'descending'
    if _same(S[1], P[1]):
        return True
    elif deg_in(R, S[1], R.leading_derivative(S[1])) == 1:
        return True
    return False


def num(R, f, pair):
    """Maple num(f, ZD[i]): which element of the pair equals f (1 or 2), else False."""
    if _same(f, pair[0]):
        return 1
    elif _same(f, pair[1]):
        return 2
    return False


def Sec_Crit(R, f, g, ZD, parms):
    """Maple Sec_Crit."""
    if not ZD:
        return False
    i = 0
    while i < len(ZD):
        n = num(R, f, ZD[i])
        if n is False:
            i += 1
            continue
        j = 1 if n == 1 else 0
        # member({g, ZD[i][j]}, ZD)
        target = {sympy.srepr(expand(g)), sympy.srepr(expand(ZD[i][j]))}
        present = any({sympy.srepr(expand(z[0])), sympy.srepr(expand(z[1]))} == target
                      for z in ZD)
        if present:
            if SC(R, [f, ZD[i][j], g], parms) is True:
                return True
            else:
                i += 1
        else:
            i += 1
    return False


# ---------------------------------------------------------------------------
# update  (paper Algorithm 4.6 / par-rga update)
# ---------------------------------------------------------------------------

def update(R, h, Gold, Bold, parms):
    """Port of par-rga ``update``: add h to the chain, refresh critical pairs.

    Returns (Gnew, Ex, Bnew):
        Gnew -- new processed-poly set (with h)
        Ex   -- members of Gold that became reducible w.r.t h (to reprocess)
        Bnew -- new critical-pair set
    """
    f = SIM(h)
    D = []
    E = []
    C = set()
    K = []
    d = factor_deriv(R, R.leading_derivative(f))
    Gnew = list(Gold)
    Ex = []
    for g in list(Gold):
        if leaderreduced(R, g, f):
            Gnew = Extract(Gnew, [g])
            K.append(g)
        elif isreduceble(R, g, f):
            Ex.append(g)
            Gnew = Extract(Gnew, [g])
    DD = ADD(Gnew, K)
    Cset = []
    # A coordinate relation (independent-variable leader) forms no differential
    # critical pairs: its "leader" is not a derivative, so DeltaPolynomial ->
    # theta_symbol -> R.factor_derivative raises 'dependent variable expected'.
    # Skip pairs where either side has an independent-variable leader (same
    # principle as the isreduceble/leaderreduced guards).
    if R.leading_derivative(f) not in _indep(R):
        for dd in DD:
            ld_dd = R.leading_derivative(SIM(dd))
            if ld_dd not in _indep(R) and _same(factor_deriv(R, ld_dd)[1], d[1]):
                pair = (SIM(dd), f)
                Cset.append(pair)
    T = Test(R, f, parms)
    Cwork = list(Cset)
    while Cwork:
        s = Cwork.pop(0)
        hd = factor_deriv(R, R.leading_derivative(SIM(s[1])))
        if T and Test(R, s[1], parms) and gcd(hd[0], d[0]) == 1:
            D.append(s)
        elif (Sec_Crit(R, f, s[1], Cwork, parms) is False or
              Sec_Crit(R, f, s[1], D, parms) is False):
            D.append(s)
            E.append(s)
    Bnew = []
    B = list(Bold)
    while B:
        k = B[0]
        B = Extract(B, [k])
        ldk1 = factor_deriv(R, R.leading_derivative(SIM(k[0])))[0]
        ldk2 = factor_deriv(R, R.leading_derivative(SIM(k[1])))[0]
        if (SC(R, [k[0], f, k[1]], parms) is False or
                gcd(ldk1, ldk2) == gcd(ldk1, d[0]) or
                gcd(ldk1, ldk2) == gcd(ldk2, d[0])):
            Bnew.append(k)
    Bnew = Bnew + E
    Gnew = list(Gnew) + [f]
    return Gnew, Ex, Bnew
