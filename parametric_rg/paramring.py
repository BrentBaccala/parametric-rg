"""Parametric (N, W) side: Groebner / ideal work over the parameter ring.

This is the analogue of par-rga.txt's
``Groebner[NormalForm](., N, plex(parms))``, ``Groebner[Basis]``,
``RadicalMembership``, ``IdealMembership`` and ``EliminationIdeal`` calls.
It uses Sage/Singular over QQ[parms] with a lex ordering on the parameters.

Must be imported inside a Sage session (`from sage.all import *`).  Use
Sage-safe syntax only (``**`` not ``^``, explicit ``QQ``/``PolynomialRing``).
"""

from sage.all import QQ, PolynomialRing, ideal, Integer as SageInteger
import sympy


def _qq(r):
    """Convert a sympy rational/integer to a Sage rational."""
    return QQ(str(sympy.Rational(r)))


class ParamRing(object):
    """A polynomial ring QQ[parms] with lex order, bridging sympy <-> Sage."""

    def __init__(self, parms):
        # parms: list of sympy Symbols
        self.parms = list(parms)
        self.names = [str(p) for p in self.parms]
        # lex ordering on the parameters (par-rga uses plex(op(parms)))
        self.R = PolynomialRing(QQ, self.names, order='lex')
        self.gens = {n: g for n, g in zip(self.names, self.R.gens())}
        # sympy symbols keyed by name
        self.sym = {str(p): p for p in self.parms}

    # ---- conversions -------------------------------------------------
    def to_sage(self, expr):
        """Convert a sympy polynomial in the parameters to a Sage poly."""
        expr = sympy.sympify(expr)
        expr = sympy.expand(expr)
        if expr.free_symbols and not (expr.free_symbols <= set(self.parms)):
            raise ValueError("to_sage: expression involves non-parameters: %s"
                             % (expr.free_symbols - set(self.parms)))
        # use sympy's polynomial dict then rebuild in Sage to avoid string parsing
        if expr == 0:
            return self.R(0)
        if not expr.free_symbols:
            return self.R(_qq(expr))
        poly = sympy.Poly(expr, *self.parms)
        result = self.R(0)
        for monom, coeff in poly.terms():
            term = self.R(_qq(coeff))
            for g, e in zip(self.R.gens(), monom):
                term = term * g**int(e)
            result = result + term
        return result

    def to_sympy(self, sage_poly):
        """Convert a Sage poly over QQ[parms] back to a sympy expression."""
        expr = sympy.Integer(0)
        for coeff, monom in zip(sage_poly.coefficients(), sage_poly.monomials()):
            term = sympy.Rational(str(coeff))
            ed = monom.exponents()[0] if hasattr(monom, 'exponents') else None
            # robust: use monomial.degrees()
            degs = monom.degrees()
            for sym, e in zip(self.parms, degs):
                term = term * sym**int(e)
            expr = expr + term
        return sympy.expand(expr)

    # ---- Groebner / ideal operations ---------------------------------
    def groebner_basis(self, polys):
        """Reduced lex Groebner basis of <polys> over QQ[parms] (as sympy list).

        Analogue of Groebner[Basis](L, plex(parms))."""
        sage_polys = [self.to_sage(p) for p in polys if sympy.sympify(p) != 0]
        if not sage_polys:
            return []
        I = self.R.ideal(sage_polys)
        gb = I.groebner_basis()
        return [self.to_sympy(g) for g in gb]

    def normal_form(self, f, N):
        """Normal form of f modulo the ideal <N> (lex).  Groebner[NormalForm].

        N may be a list of generators (a basis) or polynomials; we reduce f by
        the Groebner basis of <N>."""
        f = sympy.sympify(f)
        if f == 0:
            return sympy.Integer(0)
        # If f involves non-parameters, only the parameter part can be reduced;
        # par-rga only ever calls NormalForm on parameter polynomials here.
        if f.free_symbols and not (f.free_symbols <= set(self.parms)):
            # treat the differential variables as additional "coefficients":
            # reduce coefficient-wise is not what Maple does; Maple's NormalForm
            # over plex(parms) treats other indeterminates as coefficients.
            return self._normal_form_mixed(f, N)
        sage_f = self.to_sage(f)
        Nlist = [self.to_sage(p) for p in N if sympy.sympify(p) != 0]
        if not Nlist:
            return sympy.expand(f)
        I = self.R.ideal(Nlist)
        r = I.reduce(sage_f)
        return self.to_sympy(r)

    def _normal_form_mixed(self, f, N):
        """NormalForm where f also involves differential variables.

        Build a bigger ring QQ[diffvars][parms] and reduce.  We reduce the
        parameter-coefficients of f modulo <N>.  Implementation: collect f as a
        polynomial in the differential variables, reduce each coefficient
        (which lives in QQ[parms]) modulo <N>."""
        f = sympy.expand(f)
        diffvars = sorted(f.free_symbols - set(self.parms), key=str)
        if not diffvars:
            return self.normal_form(f, N)
        Nlist = [self.to_sage(p) for p in N if sympy.sympify(p) != 0]
        if not Nlist:
            return f
        I = self.R.ideal(Nlist)
        poly = sympy.Poly(f, *diffvars)
        result = sympy.Integer(0)
        for monom, coeff in poly.terms():
            csym = sympy.expand(coeff)
            if csym.free_symbols <= set(self.parms):
                cred = self.to_sympy(I.reduce(self.to_sage(csym)))
            else:
                cred = csym
            term = cred
            for v, e in zip(diffvars, monom):
                term = term * v**int(e)
            result = result + term
        return sympy.expand(result)

    def ideal_membership(self, f, N):
        """True if f in <N> (lex Groebner).  Analogue of IdealMembership."""
        f = sympy.sympify(f)
        Nlist = [self.to_sage(p) for p in N if sympy.sympify(p) != 0]
        if not Nlist:
            return sympy.sympify(f) == 0
        I = self.R.ideal(Nlist)
        return I.reduce(self.to_sage(f)) == self.R(0)

    def radical_membership(self, f, N):
        """True if f in radical(<N>).  Analogue of RadicalMembership.

        Uses the Rabinowitsch trick: f in sqrt(I) iff 1 in I + <1 - t f> in
        QQ[parms, t]."""
        f = sympy.sympify(f)
        Nlist = [p for p in N if sympy.sympify(p) != 0]
        if not Nlist:
            return sympy.sympify(f) == 0
        names = self.names + ['t_rabin']
        Rt = PolynomialRing(QQ, names, order='lex')
        gens = {n: g for n, g in zip(names, Rt.gens())}
        t = gens['t_rabin']

        def emb(expr):
            expr = sympy.expand(sympy.sympify(expr))
            if expr == 0:
                return Rt(0)
            if not expr.free_symbols:
                return Rt(_qq(expr))
            poly = sympy.Poly(expr, *self.parms)
            res = Rt(0)
            for monom, coeff in poly.terms():
                term = Rt(_qq(coeff))
                for nm, e in zip(self.names, monom):
                    term = term * gens[nm]**int(e)
                res = res + term
            return res

        gens_list = [emb(p) for p in Nlist] + [1 - t * emb(f)]
        I = Rt.ideal(gens_list)
        return Rt(1) in I

    def is_trivial(self, N):
        """True if <N> is the whole ring (i.e. 1 in <N>); GB == [1]."""
        gb = self.groebner_basis(N)
        return len(gb) == 1 and sympy.sympify(gb[0]).is_constant() and gb[0] != 0
