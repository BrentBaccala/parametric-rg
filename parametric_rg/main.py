"""Parametric Rosenfeld-Groebner (Fakouri-Rahmany-Basiri 2018) -- main driver.

Faithful port of ~/project/rahmani/par-rga.txt, cross-referenced against
section 4 of the paper (Algorithms 4.1-4.8).  Built on the stock Python
DifferentialAlgebra binding (with DeltaPolynomial reconstructed in diffprim).

Vertex layout (par-rga's 7-list ``[V1,V2,V3,V4,V5,V6,V7]``), which corresponds
to the paper's sextuplet <A, D, P, S, N, W> as:

    V[0] = A   processed polynomials (the regular chain so far)
    V[1] = P   non-processed polynomials (input equations still to handle)
    V[2] = D   critical pairs (delta set) of A still to solve
    V[3] = S   inequations / non-zero differential conditions
    V[4] = S2  extra non-zero differential conditions (separant/initial)
    V[5] = N   zero parametric conditions (list, kept as a lex Groebner basis)
    V[6] = W   non-zero parametric conditions (list)

Must be imported inside a Sage session (it pulls in paramring, which needs
sage.all).  Run via ~/miniforge3/envs/sage/bin/sage -python.
"""

import sympy
from sympy import expand, S, Integer, Poly, gcd

from .util import (GCD, DIVIDE, RZ, SIM, ADD, Extract, member, PComponent,
                   FACTOR, _same, is_constant)
from .diffprim import DeltaPolynomial
from .diffhelpers import (update, factor_deriv, leaderreduced, isreduceble,
                          deg_in)
from .paramring import ParamRing


# vertex index names for readability
A, P, D, Sineq, S2, N, W = 0, 1, 2, 3, 4, 5, 6


class PRG(object):
    """Parametric Rosenfeld-Groebner solver bound to a ring R and parameters."""

    def __init__(self, R, derivations, parms, verbose=False):
        self.R = _RingWrap(R, list(derivations))
        self.derivations = list(derivations)
        self.parms = list(parms)
        self.PR = ParamRing(self.parms)
        self.verbose = verbose

    # -- small ring-bound conveniences --------------------------------
    def ld(self, p):
        return self.R.leading_derivative(p)

    def init(self, p):
        return self.R.initial(p)

    def sep(self, p):
        return self.R.separant(p)

    def degld(self, p):
        return deg_in(self.R, p, self.ld(p))

    def diffprem(self, p, redset, mode='full'):
        """ParmDiffRed: differential pseudo-remainder of p by redset.

        par-rga adjoins the parameters' derivatives ``parms[j][vars[i]]`` to the
        reduction set.  On the stock Python binding the parameters are genuine
        constants, so their derivatives are literally 0 and contribute nothing;
        we therefore drop them (and any other zero) from the redset.  The
        reduction by the chain itself is unchanged."""
        full_redset = [r for r in redset if sympy.sympify(r) != 0]
        if not full_redset:
            return expand(sympy.sympify(p))
        h, r = self.R.differential_prem(p, full_redset, mode)
        return r

    def nf(self, f, Nlist):
        """Groebner[NormalForm](f, N, plex(parms))."""
        if not Nlist:
            return expand(sympy.sympify(f))
        return self.PR.normal_form(f, Nlist)

    # -----------------------------------------------------------------
    # NewPCondition  (paper Find-New-ParCondition, Algorithm 4.4)
    # -----------------------------------------------------------------
    def NewPCondition(self, p, Nlist, Wlist):
        """Return parametric (zero) conditions of p w.r.t. (N, W).

        Faithful port of par-rga NewPCondition.  Returns one of:
            (False, p)                 -- p is purely parametric
            (False, [])                -- no parametric condition
            (False, [See])             -- separant factor lies in radical(N)
                                          (=> p cannot vanish here; reprocess)
            (True, par, r)             -- one new condition, r in {'i','s'}
            (True, par)                -- two new conditions [sep_fac, init_fac]
        """
        parmset = set(self.parms)
        par = []
        # purely-parametric polynomial
        ind = (sympy.sympify(p).free_symbols | sympy.sympify(p).atoms(sympy.Derivative))
        if ind <= parmset:
            return (False, p)

        See = None
        Se = SIM(self.sep(p))
        See = FACTOR(Se, parmset)
        r = None
        if See is not False:
            if (not Nlist) or (self.PR.radical_membership(See, Nlist) is False):
                See2 = DIVIDE(See, Wlist)
                if not is_constant(See2):
                    r = 's'
                    par = [See2]
                See = See2
            else:
                par = False

        if par is not False:
            In = SIM(self.init(p))
            Ini = FACTOR(In, parmset)
            if Ini is not False:
                Ini = DIVIDE(Ini, Wlist)
                if not is_constant(Ini):
                    r = 'i'
                    if not is_constant(DIVIDE(Ini, par if isinstance(par, list) else [par])):
                        par = ADD([Ini], par if isinstance(par, list) else [])

        if _same_obj(par, p):
            return (False, p)
        elif par == []:
            return (False, [])
        elif par is False:
            return (False, [See])
        elif len(par) == 1:
            return (True, par, r)
        else:
            return (True, par)

    # -----------------------------------------------------------------
    # NewCondition  (paper Find-New-DiffCondition, Algorithm 4.5)
    # -----------------------------------------------------------------
    def NewCondition(self, f, oc, pcond, NZP):
        """Differential zero/non-zero conditions of f  (par-rga NewCondition).

        oc   -- old conditions (the P[4]=Sineq inequation set)
        pcond-- current non-zero conditions P[5]=S2
        NZP  -- non-zero parametric conditions P[7]=W
        Returns [False, F] or [True, E, F, e].
        """
        OC1 = ADD(oc, pcond)
        OC = ADD(OC1, NZP)
        E = []
        Fset = list(pcond)
        e = None
        inn = SIM(self.init(f))
        s = SIM(self.sep(f))
        initv = DIVIDE(inn, OC)
        if not is_constant(initv):
            # if no h in OC1 reduces initv to 0 -> it is a genuine non-zero cond
            reduced_to_zero = any(self.diffprem(initv, [oc1]) == 0 for oc1 in OC1) \
                if False else False
            # par-rga checks member(0, [ParmDiffRed(OC1[i], [init], full)]):
            reduced = any(self.diffprem(oc1, [initv]) == 0 for oc1 in OC1)
            if not reduced:
                E = [initv]
                OC1 = ADD(OC1, [initv])
                e = 'i'
            else:
                Fset = Fset + [initv]
        sepv = DIVIDE(expand(s), OC)
        if not is_constant(sepv):
            reduced = any(self.diffprem(oc1, [sepv]) == 0 for oc1 in OC1)
            if not reduced:
                E = ADD(E, [sepv])
                e = 's'
            elif not _same(initv, sepv):
                Fset = Fset + [sepv]
        if not E:
            return [False, Fset]
        else:
            return [True, E, Fset, e]

    # -----------------------------------------------------------------
    # Branch  (par-rga Branch -- the differential branching, Make-Tree II core)
    # -----------------------------------------------------------------
    def Branch(self, V, p):
        """Branch vertex V on differential conditions of p.  Returns list of
        new vertices.  Faithful port of par-rga Branch."""
        B = V[D]
        f = _strip_power(sympy.factor(p))
        out = []
        if f.func != sympy.Add:
            h = SIM(f)
            if h.func == sympy.Mul:
                # product of distinct factors -> branch by zeroing each factor
                O = list(h.args)
                for o in O:
                    if is_constant(o):
                        continue
                    o = _strip_power(o)
                    out.append(_mk(V, P, ADD([o], V[P])))
                return out
            else:
                Gnew, Ex, Bnew = update(self.R, f, V[A], B, self.parms)
                newv = list(V)
                newv[A] = RZ(Gnew)
                newv[P] = ADD(Ex, Extract(V[P], [f, -f]))
                newv[D] = Bnew
                return [newv]
        else:
            NC = self.NewCondition(f, V[Sineq], V[S2], V[W])
            Z = []
            if NC[0] is False:
                Z = NC[1]
                Gnew, Ex, Bnew = update(self.R, f, V[A], B, self.parms)
                TR = [DIVIDE(g, NC[1]) for g in Gnew]
                newv = list(V)
                newv[A] = RZ(TR)
                newv[P] = ADD(Ex, Extract(V[P], [f, -f]))
                newv[D] = Bnew
                newv[S2] = Z
                return [newv]
            elif len(NC[1]) == 1:
                Gnew, Ex, Bnew = update(self.R, f, V[A], B, self.parms)
                TR = [DIVIDE(g, NC[1]) for g in Gnew]
                Z = ADD(Z, NC[2])
                Z3 = ADD([], NC[2])
                # build the "reduced" companion f1 by dropping init or sep term
                if NC[3] == 'i':
                    f1 = expand(f - self.ld(f)**self.degld(f) * self.init(f))
                else:
                    f1 = expand(self.degld(f) * f - self.ld(f) * self.sep(f))
                v1 = list(V)
                v1[A] = TR
                v1[P] = ADD(Ex, Extract(V[P], [f, -f]))
                v1[D] = Bnew
                v1[Sineq] = ADD(V[Sineq], [NC[1][0]])
                v1[S2] = Z3
                v2 = list(V)
                v2[P] = RZ(ADD(Extract(V[P], [f, -f]), [NC[1][0], SIM(f1)]))
                v2[D] = Extract(V[D], [f, -f])
                v2[S2] = Z
                return [v1, v2]
            else:
                # two conditions
                P4 = list(V[Sineq])
                Z2 = []
                for cond in list(V[Sineq]):
                    if self.diffprem(NC[1][0], [cond]) == 0:
                        P4 = Extract(P4, [cond])
                        Z = ADD(Z, [cond])
                        Z2 = ADD(Z2, [cond])
                    elif self.diffprem(NC[1][1], [cond]) == 0:
                        P4 = Extract(P4, [cond])
                        Z = ADD(Z, [cond])
                f1 = expand(f - self.ld(f)**self.degld(f) * self.init(f))
                g = expand(self.degld(f) * f - self.ld(f) * self.sep(f))
                Gnew, Ex, Bnew = update(self.R, f, V[A], B, self.parms)
                TR = [DIVIDE(gg, NC[1]) for gg in Gnew]
                Z = ADD(Z, NC[2])
                Z2 = ADD(Z2, NC[2])
                v1 = list(V)
                v1[A] = RZ(TR)
                v1[P] = ADD(Ex, Extract(V[P], [f, -f]))
                v1[D] = Bnew
                v1[Sineq] = ADD(P4, NC[1])
                v1[S2] = Z
                v2 = list(V)
                v2[P] = RZ(ADD(Extract(V[P], [f, -f]), [NC[1][0], SIM(f1)]))
                v2[D] = Extract(V[D], [f, -f])
                v2[S2] = Z
                v3 = list(V)
                v3[P] = RZ(ADD(Extract(V[P], [f, -f]), [NC[1][1], SIM(g)]))
                v3[D] = Extract(V[D], [f, -f])
                v3[Sineq] = ADD(P4, [NC[1][0]])
                v3[S2] = Z2
                return [v1, v2, v3]

    # -----------------------------------------------------------------
    # MakeTree  (paper Make-Tree I, Algorithm 4.2)
    # -----------------------------------------------------------------
    def MakeTree(self, p, V):
        """Branch vertex V w.r.t the parametric & differential conditions of p."""
        New = []
        NC = self.NewPCondition(p, V[N], V[W])
        if NC[0] is False and _same_obj(NC[1], p):
            # p purely parametric: it is a zero parametric condition.
            PC = PComponent(p)
            P1 = list(V[A])
            P2 = list(V[P])
            for comp in PC:
                for poly in list(V[A]):
                    if not _same(self.nf(poly, [comp]), poly):
                        P1 = Extract(P1, [poly])
                        P2 = ADD([poly], P2)
                newv = list(V)
                newv[A] = P1
                newv[P] = P2
                newv[N] = ADD(V[N], [comp])
                New.append(newv)
            return New
        elif NC[0] is False and NC[1] == []:
            return self.Branch(V, p)
        elif NC[0] is False:
            # separant factor in radical(N): p reduces to its lower part
            q = expand(self.degld(p) * p - self.ld(p) * self.sep(p))
            newv = list(V)
            newv[P] = ADD([q], V[P])
            return [newv]
        elif NC[0] is True and len(NC[1]) == 1:
            if NC[2] == 'i':
                q = expand(p - self.ld(p)**self.degld(p) * self.init(p))
            else:
                q = expand(self.degld(p) * p - self.ld(p) * self.sep(p))
            PC = PComponent(NC[1][0])
            P1 = list(V[A])
            P2 = list(V[P])
            for comp in PC:
                for poly in list(V[A]):
                    if not _same(self.nf(poly, [comp]), poly):
                        P1 = Extract(P1, [poly])
                        P2 = ADD([poly], P2)
                newv = list(V)
                newv[A] = P1
                newv[P] = ADD([q], P2)
                newv[N] = ADD(V[N], [comp])
                New.append(newv)
            # plus the non-zero branch: Branch with the condition in W
            Vb = list(V)
            Vb[W] = ADD(V[W], PC)
            New = New + self.Branch(Vb, DIVIDE(p, PC))
            return New
        else:
            # two parametric conditions
            q1 = expand(p - self.ld(p)**self.degld(p) * self.init(p))
            q2 = expand(self.degld(p) * p - self.ld(p) * self.sep(p))
            PC = PComponent(NC[1][0])
            PC2 = PComponent(NC[1][1])
            P1 = list(V[A])
            P2 = list(V[P])
            for comp in PC:
                for poly in list(V[A]):
                    if not _same(self.nf(poly, [comp]), poly):
                        P1 = Extract(P1, [poly])
                        P2 = ADD([poly], P2)
                newv = list(V)
                newv[A] = P1
                newv[P] = ADD([q1], P2)
                newv[N] = ADD(V[N], [comp])
                New.append(newv)
            New1 = []
            for comp in PC2:
                P12 = [self.nf(poly, [comp]) for poly in V[A]]
                newv = list(V)
                newv[A] = P12
                newv[P] = ADD([q2], V[P])
                newv[N] = ADD(V[N], [comp])
                newv[W] = ADD(V[W], PC)
                New1.append(newv)
            Vb = list(V)
            Vb[W] = ADD(ADD(V[W], PC), PC2)
            tail = self.Branch(Vb, DIVIDE(p, [c for c in PC] + [c for c in PC2]))
            return New + New1 + tail

    # -----------------------------------------------------------------
    # CheckParm  (paper Parametric-Consistency, Algorithm 4.7)
    # -----------------------------------------------------------------
    def CheckParm(self, Pol, Zp, Nzp):
        """Return (N, Nzp, ZP) or False  (par-rga CheckParm)."""
        parmset = set(self.parms)
        ZP = []
        for dd in Pol:
            dd = sympy.sympify(dd)
            if is_constant(dd):
                return False
            ind = dd.free_symbols | dd.atoms(sympy.Derivative)
            if ind <= parmset:
                ZP = ADD([dd], ZP)
        Nbasis = self.PR.groebner_basis(ADD(ZP, Zp))
        if len(Nbasis) == 1 and is_constant(Nbasis[0]) and Nbasis[0] != 0:
            return False
        # consistency of non-zero conditions: prod(Nzp) not in <N>
        if not Nzp:
            return (Nbasis, Nzp, ZP)
        h = Integer(1)
        for z in Nzp:
            h = h * z
        if self.PR.ideal_membership(h, Nbasis) is False:
            return (Nbasis, Nzp, ZP)
        return False

    # -----------------------------------------------------------------
    # CheckBranch  (paper Differential-Consistency, Algorithm 4.8)
    # -----------------------------------------------------------------
    def CheckBranch(self, V):
        """Return False (inconsistent) or a refined vertex.  par-rga CheckBranch."""
        C = self.CheckParm(V[A], V[N], V[W])
        if C is False:
            return False
        Nbasis, Nzp, ZP = C
        P1 = Extract(V[A], ZP)
        P2 = RZ(V[P])
        # 0 among inequations or constant equation => inconsistent
        if any(is_constant(x) for x in P2):
            return False
        parmset = set(self.parms)
        if not V[Sineq] and not V[S2]:
            newv = list(V)
            newv[A] = P1
            newv[P] = P2
            newv[N] = Nbasis
            newv[W] = Nzp
            return newv
        else:
            Z = [self.nf(s, Nbasis) for s in V[Sineq]]
            L = []
            for i in range(len(V[Sineq])):
                red = SIM(expand(self.diffprem(Z[i], P1)))
                divisors = [Z[j] for j in range(i + 1, len(Z))] + list(V[W])
                L.append(DIVIDE(red, divisors))
            if member(0, L):
                return False
            L1 = list(L)
            Bextra = []
            for li in L:
                if is_constant(li):
                    L1 = Extract(L1, [li])
                elif (sympy.sympify(li).free_symbols |
                      sympy.sympify(li).atoms(sympy.Derivative)) <= parmset:
                    L1 = Extract(L1, [li])
                    Bextra = ADD([li], Bextra)
                else:
                    g = FACTOR(li, parmset)
                    if g is not False:
                        L1 = ADD(Extract(L1, [li]), [expand(sympy.cancel(li / g))])
                        Bextra = ADD(PComponent(g), Bextra)
            newv = list(V)
            newv[A] = P1
            newv[P] = P2
            newv[Sineq] = L1
            newv[N] = Nbasis
            newv[W] = ADD(Nzp, Bextra)
            return newv

    # -----------------------------------------------------------------
    # checkregular  (Rosenfeld's-lemma consistency filter)
    # -----------------------------------------------------------------
    def checkregular(self, decom):
        """Keep the consistent regular systems (par-rga checkregular).

        Each element of decom is [A_list, S_list, N_list, W_list].  We test
        1 not in (A) : S^inf  via the saturation/elimination ideal trick, using
        a fresh polynomial ring with the diff-derivatives turned into algebraic
        indeterminates.
        """
        from sage.all import QQ, PolynomialRing
        out = []
        for entry in decom:
            Aset, Sset, Nset, Wset = entry
            b1 = list(Aset) + list(Sset) + list(Nset) + list(Wset)
            # collect the algebraic indeterminates: derivative jets, dependent
            # variable applications, and parameter symbols.  The independent
            # variables x,y,z appear only as arguments and are NOT indeterminates.
            indt = set()
            for poly in b1:
                e = sympy.sympify(poly)
                indt |= _algebraic_indets(e, self.parms, self.derivations)
            indt = sorted(indt, key=str)
            if not indt:
                out.append(entry)
                continue
            # map each indeterminate to a fresh algebraic variable
            names = ['X%d' % i for i in range(len(indt))]
            subsmap = {ind: sympy.Symbol(nm) for ind, nm in zip(indt, names)}
            Slist = [sympy.sympify(s).xreplace(subsmap) for s in (list(Sset) + list(Wset))]
            gnames = ['g%d' % i for i in range(len(Slist))]
            allnames = names + gnames
            Rr = PolynomialRing(QQ, allnames, order='lex') if allnames else PolynomialRing(QQ, ['x_dummy'])
            gens = {nm: g for nm, g in zip(allnames, Rr.gens())}

            def conv(expr):
                expr = sympy.expand(sympy.sympify(expr).xreplace(subsmap))
                return _sympy_to_sage(expr, Rr, gens)

            polysC = [conv(a) for a in (list(Aset) + list(Nset))]
            for k, s in enumerate(Slist):
                polysC.append(gens['g%d' % k] * conv(s) - 1)
            if not polysC:
                out.append(entry)
                continue
            I = Rr.ideal(polysC)
            # eliminate the g-variables: 1 in elimination ideal w.r.t the N[i]?
            # par-rga tests 1 not in EliminationIdeal(<C>, {N_i}) where N_i are the
            # diff-variables.  Equivalent: the system C has no solution => 1 in I.
            if Rr(1) not in I:
                out.append(entry)
        return out

    # -----------------------------------------------------------------
    # MainProc  (Algorithm 4.1)
    # -----------------------------------------------------------------
    def MainProc(self, Pin, Qin, max_vertices=20000):
        """Run the parametric RG on equations Pin (=0) and inequations Qin (!=0).

        Returns the list of consistent regular representations, each as
        [A_list, S_list, N_list, W_list]."""
        Decom = []
        NP = []
        root = [[], list(Pin), [], list(Qin), [], [], []]
        Br = [root]
        count = 0
        while Br:
            count += 1
            if count > max_vertices:
                raise RuntimeError("vertex budget exceeded (%d)" % max_vertices)
            cur = Br[0]
            Br = Br[1:]
            T = self.CheckBranch(cur)
            if T is False:
                NP.append(cur)
                continue
            V = T
            if V[D]:
                # solve a critical pair via its DeltaPolynomial
                pair = V[D][0]
                q = DeltaPolynomial(self.R, pair[0], pair[1], self.derivations)
                # par-rga reduces q by the parameter-derivatives first; those are
                # 0 on the stock binding (parameters are constants), so skip.
                q = expand(self.diffprem(q, V[A]))
                if V[N]:
                    q = self.nf(q, V[N])
                q = SIM(q)
                if is_constant(q) and q != 0:
                    NP.append(V)
                elif q == 0:
                    nv = list(V)
                    nv[D] = Extract(V[D], [V[D][0]])
                    Br = [nv] + Br
                else:
                    q = DIVIDE(q, list(ADD(V[S2], V[Sineq])) + list(V[W]))
                    if is_constant(q):
                        NP.append(V)
                    else:
                        v1 = list(V)
                        v1[D] = Extract(V[D], [V[D][0]])
                        Br4 = self.MakeTree(q, v1)
                        Br = list(Br4) + Br
            elif V[P]:
                q = V[P][0]
                if V[N]:
                    q = self.nf(q, V[N])
                q = expand(self.diffprem(q, V[A]))
                q = SIM(q)
                if is_constant(q) and q != 0:
                    NP.append(V)
                elif q == 0:
                    nv = list(V)
                    nv[P] = Extract(V[P], [V[P][0]])
                    Br = [nv] + Br
                else:
                    q = DIVIDE(q, list(ADD(V[S2], V[Sineq])) + list(V[W]))
                    if is_constant(q):
                        NP.append(V)
                    else:
                        v1 = list(V)
                        v1[P] = Extract(V[P], [V[P][0]])
                        Br4 = self.MakeTree(q, v1)
                        Br = list(Br4) + Br
            else:
                Decom.append(V)
        # assemble [A, S+S2, N, W] for each terminal vertex
        systems = [[V[A], ADD(V[Sineq], V[S2]), V[N], V[W]] for V in Decom]
        de = self.checkregular(systems)
        return de


# ===========================================================================
# module-level helpers
# ===========================================================================

class _RingWrap(object):
    """Thin wrapper so we can attach ``_derivations`` (the compiled
    DifferentialRing forbids setting attributes) and forward everything else."""

    def __init__(self, R, derivations):
        object.__setattr__(self, '_R', R)
        object.__setattr__(self, '_derivations', list(derivations))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_R'), name)


def _strip_power(f):
    """If f is base**k, return base; else f.  (Maple `if type(f,^) then [op(f)][1]`)."""
    f = sympy.sympify(f)
    if f.func == sympy.Pow:
        return f.args[0]
    # factor() may yield Mul with a Pow; par-rga only strips a top-level Pow.
    return f


def _mk(V, idx, value):
    nv = list(V)
    nv[idx] = value
    return nv


def _same_obj(a, b):
    """Structural equality used where par-rga compares `par = p`."""
    try:
        return expand(sympy.sympify(a) - sympy.sympify(b)) == 0
    except Exception:
        return a is b or a == b


def _algebraic_indets(expr, parms, derivations):
    """The algebraic indeterminates of a differential polynomial: derivative
    jets, dependent-variable applications, and parameter symbols.  Independent
    variables (the derivations) are excluded -- they only appear as arguments."""
    expr = sympy.sympify(expr)
    indt = set()
    indt |= expr.atoms(sympy.Derivative)
    for f in expr.atoms(sympy.Function):
        indt.add(f)
    for s in expr.free_symbols:
        if s in set(parms):
            indt.add(s)
    # drop the independent variables if any leaked in as bare symbols
    indt = {x for x in indt if x not in set(derivations)}
    return indt


def _sympy_to_sage(expr, Rr, gens):
    """Convert a sympy polynomial (already substituted to X-names) to Sage."""
    from sage.all import QQ
    expr = sympy.expand(expr)
    if expr == 0:
        return Rr(0)
    syms = sorted(expr.free_symbols, key=str)
    if not syms:
        return Rr(QQ(str(sympy.Rational(expr))))
    poly = sympy.Poly(expr, *syms)
    res = Rr(0)
    for monom, coeff in poly.terms():
        term = Rr(QQ(str(sympy.Rational(coeff))))
        for sym, e in zip(syms, monom):
            term = term * gens[str(sym)]**int(e)
        res = res + term
    return res
