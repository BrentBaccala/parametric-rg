"""
Differential-algebra primitives for the parametric Rosenfeld-Groebner port.

This module wraps Boulier's Python ``DifferentialAlgebra`` binding (BLAD/BMI)
and reconstructs the one primitive the Python binding does NOT expose,
``DeltaPolynomial`` (it exists in BLAD and in the Maple binding only).

The reconstruction mirrors the BLAD C implementation
``bad_delta_polynomial_critical_pair`` in
``~/DifferentialAlgebra/blad/bad/src/bad_critical_pair.c``:

  Let u1, u2 be the leading derivatives of p1, p2 and u12 their least common
  derivative (the derivative of the common dependent variable whose derivation
  operator is the lcm of theta(u1), theta(u2)).

  * Triangular case (u12 != u1 and u12 != u2):
        theta_i      = operator carrying u_i  -> u12
        PP_i         = (theta_i) applied to p_i          (leader becomes u12)
        s_i          = separant of p_i
        Delta        = reductum(PP_1)*s_2 - reductum(PP_2)*s_1
    BLAD divides s_1,s_2 by their gcd first; we replicate that, then
    normalise the numeric content of Delta.  The leading-u12 terms of the
    two differentiated polynomials are s_i*u12 and cancel, so taking the
    reductum (dropping the u12 term) and cross-multiplying by the separants
    is exactly the classical Delta-polynomial
        s1*(theta12/theta2)*p2 - s2*(theta12/theta1)*p1
    restricted to the part below u12.

  * Non-triangular case (u12 == u1 or u12 == u2): one leader is a derivative
    of the other.  BLAD orders the pair so P1 has higher rank, differentiates
    P2 up to u12 if needed, and takes a (gcd-)pseudo-remainder of P1 by the
    (differentiated) P2 w.r.t. u12.  We replicate that with the binding's own
    ``differential_prem`` (Ritt reduction), which performs exactly this
    pseudo-remainder.

Validated against the BMI test oracle
``~/DifferentialAlgebra/bmi/maple/tests/delta_polynomial.tst``.
"""

import sympy
from sympy import Derivative, Poly, expand, gcd, primitive, S, Integer


# ---------------------------------------------------------------------------
# small helpers on derivatives / theta operators
# ---------------------------------------------------------------------------

def theta_symbol(R, deriv):
    """Return (theta_monomial, symb) for a derivative, via factor_derivative.

    ``theta_monomial`` is a monomial in the independent variables (1 for an
    order-0 derivative / a parameter); ``symb`` is the underlying dependent
    variable (e.g. ``u(x, y)``)."""
    theta, symb = R.factor_derivative(deriv)
    return sympy.sympify(theta), symb


def theta_exponents(theta, derivations):
    """Exponent vector of a theta monomial over the derivation variables."""
    p = Poly(theta, *derivations)
    if theta == 1 or p.is_zero:
        return tuple(0 for _ in derivations)
    # theta is a single monomial; take its (unique) exponent tuple
    monoms = p.monoms()
    return monoms[0]


def lcd_variable(R, u1, u2, derivations):
    """Least common derivative of two leaders u1, u2 (must share a symb).

    Returns (u12_derivative, theta12_monomial, symb).  u12 is the derivative
    of the common symbol whose theta is the lcm of theta(u1), theta(u2)."""
    t1, s1 = theta_symbol(R, u1)
    t2, s2 = theta_symbol(R, u2)
    if s1 != s2:
        raise ValueError("DeltaPolynomial: leaders are derivatives of "
                         "different dependent variables: %s vs %s" % (s1, s2))
    t12 = sympy.lcm(t1, t2)
    # build u12 by differentiating the symbol s1 by the operator t12/(t1) ...
    # easier: differentiate symb up to the exponent vector of t12.
    e12 = theta_exponents(sympy.sympify(t12), derivations)
    u12 = _diff_by_exponents(R, s1, e12, derivations)
    return u12, sympy.sympify(t12), s1


def _diff_by_exponents(R, expr, exps, derivations):
    """Differentiate ``expr`` exps[i] times w.r.t. derivations[i]."""
    args = []
    for var, e in zip(derivations, exps):
        args.extend([var] * e)
    if not args:
        return sympy.sympify(expr)
    return R.differentiate(expr, *args)


def operator_between(R, u_from, u_to, derivations):
    """Exponent vector of the operator carrying derivative u_from to u_to.

    (theta(u_to) - theta(u_from), componentwise; assumed non-negative.)"""
    tf, sf = theta_symbol(R, u_from)
    tt, st = theta_symbol(R, u_to)
    ef = theta_exponents(tf, derivations)
    et = theta_exponents(tt, derivations)
    return tuple(b - a for a, b in zip(ef, et))


# ---------------------------------------------------------------------------
# reductum w.r.t. a leader derivative
# ---------------------------------------------------------------------------

def reductum_wrt(R, poly, leader):
    """poly minus its leading-``leader`` term  =  poly - init*leader**deg.

    Uses the binding's ``initial`` (= leading coefficient w.r.t. the ring's
    leader) only when ``leader`` is the ring leader; here we project out the
    given ``leader`` explicitly with sympy so it also works for an arbitrary
    differentiated leader u12."""
    poly = expand(poly)
    p = Poly(poly, leader)
    if p.degree() <= 0:
        # does not depend on leader: reductum is the whole thing (no leading term in `leader`)
        return poly
    d = p.degree()
    lc = p.nth(d)              # coefficient of leader**d
    return expand(poly - lc * leader**d)


def coeff_of_leader_power(R, poly, leader, k):
    """Coefficient of leader**k in poly (as a sympy expression)."""
    return Poly(expand(poly), leader).nth(k)


# ---------------------------------------------------------------------------
# numeric content normalisation (gcd of coeffs = 1, leading coeff positive)
# ---------------------------------------------------------------------------

def normalize_numeric(expr):
    """Divide out the integer content; make the leading numeric coeff positive.

    Mirrors BLAD's normalisation of the Delta-polynomial's numeric
    coefficients (gcd 1, leading one positive)."""
    expr = expand(expr)
    if expr == 0:
        return S.Zero
    try:
        cont, prim = primitive(expr)
    except Exception:
        return expr
    if cont == 0:
        return expr
    # leading numeric sign: use the sign of `cont`; primitive() already makes
    # the content positive, but we also want the overall expression's leading
    # coefficient positive in the BLAD sense.  primitive() returns cont>0, so
    # the sign lives in prim. Leave as-is (gcd of integer coeffs is 1).
    return expand(prim)


# ---------------------------------------------------------------------------
# DeltaPolynomial
# ---------------------------------------------------------------------------

def DeltaPolynomial(R, p1, p2, derivations):
    """Reconstructed BLAD DeltaPolynomial(p1, p2, R).

    ``R``            -- a DifferentialRing
    ``p1``, ``p2``   -- differential polynomials whose leaders are derivatives
                        of the same dependent variable
    ``derivations``  -- the list of independent variables (ring derivations)

    Returns the Delta-polynomial, with numeric content normalised.
    """
    p1 = expand(sympy.sympify(p1))
    p2 = expand(sympy.sympify(p2))

    u1 = R.leading_derivative(p1)
    u2 = R.leading_derivative(p2)

    u12, theta12, symb = lcd_variable(R, u1, u2, derivations)

    triangular = (u12 != u1) and (u12 != u2)

    if not triangular:
        # ---- non-triangular case ----------------------------------------
        # order so that P1 has the higher leader (u1 strictly above u2, or
        # u12 == u1).  Then prem P1 by (differentiated) P2 w.r.t. u12.
        # Decide which leader is the LCD:
        if u12 == u1:
            P_high, u_high = p1, u1
            P_low,  u_low  = p2, u2
        else:  # u12 == u2
            P_high, u_high = p2, u2
            P_low,  u_low  = p1, u1
        # P_high already has leader u12. Differentiate P_low up to u12.
        if u_low == u12:
            PP_low = P_low
        else:
            exps = operator_between(R, u_low, u12, derivations)
            PP_low = _diff_by_exponents(R, P_low, exps, derivations)
        # pseudo-remainder of P_high by PP_low w.r.t. u12 (Ritt reduction).
        # differential_prem(P_high, [PP_low]) reduces P_high modulo PP_low;
        # since PP_low has leader u12 and P_high also involves u12, this is the
        # algebraic pseudo-remainder in u12.
        h, r = R.differential_prem(P_high, [PP_low])
        delta = r
    else:
        # ---- triangular case --------------------------------------------
        exps1 = operator_between(R, u1, u12, derivations)
        exps2 = operator_between(R, u2, u12, derivations)
        PP1 = _diff_by_exponents(R, p1, exps1, derivations)
        PP2 = _diff_by_exponents(R, p2, exps2, derivations)

        s1 = expand(R.separant(p1))
        s2 = expand(R.separant(p2))

        # BLAD divides the two separants by their gcd first.
        g = gcd(s1, s2)
        if g != 0 and g != 1:
            s1 = expand(sympy.cancel(s1 / g))
            s2 = expand(sympy.cancel(s2 / g))

        red1 = reductum_wrt(R, PP1, u12)
        red2 = reductum_wrt(R, PP2, u12)

        delta = expand(red1 * s2 - red2 * s1)

    return normalize_numeric(delta)
