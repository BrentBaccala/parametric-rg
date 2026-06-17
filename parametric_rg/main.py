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


class _WalledReduction(Exception):
    """Raised when a per-vertex differential reduction breaches its time/RSS
    wall (see MainProc's diffprem_wall_* knobs).  Carries ``.info`` (a dict
    with 'why' and 'elapsed')."""
    def __init__(self, info):
        super().__init__(info.get('why', 'walled'))
        self.info = info


def _child_rss_mb(pid):
    """Resident set size of pid in MiB, or None if unreadable."""
    try:
        with open('/proc/%d/status' % pid) as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024   # kB -> MiB
    except (OSError, ValueError):
        pass
    return None


# vertex index names for readability
A, P, D, Sineq, S2, N, W = 0, 1, 2, 3, 4, 5, 6
LBL = 7   # optional trailing slot: a worklist label, set when a vertex is
          # pushed onto Br (only used by the PRG_TRACE instrumentation)


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

    def _indep_leader(self, e):
        """True if e's leader is an independent variable (a derivation symbol):
        e is then a relation among the derivations with no jet content, not a
        differential equation -- it cannot act as a differential reductor, and
        the binding's differential_prem rejects it ('independent polynomial')."""
        try:
            lead = self.ld(e)
        except Exception:
            return False
        return any(lead == d for d in self.derivations)

    def _param_free(self, c):
        """True if condition c (an initial/separant) involves none of the
        parameters -- a purely differential/jet condition.  Under
        PRG_DIFF_BASEFIELD such a condition is SATURATED (kept nonzero, no
        degenerate branch), mirroring BLAD's base field; only parameter-bearing
        conditions are split into (N,W)."""
        return not (sympy.sympify(c).free_symbols & set(self.parms))

    def diffprem(self, p, redset, mode='full'):
        """ParmDiffRed: differential pseudo-remainder of p by redset.

        par-rga adjoins the parameters' derivatives ``parms[j][vars[i]]`` to the
        reduction set.  On the stock Python binding the parameters are genuine
        constants, so their derivatives are literally 0 and contribute nothing;
        we therefore drop them (and any other zero) from the redset.  We also drop
        reductors whose leader is an independent variable -- those are relations
        among the derivations (parameter conditions), carry no differential leader
        to reduce by, and make the binding throw.  The reduction by the genuine
        differential chain is unchanged."""
        full_redset = [r for r in redset
                       if sympy.sympify(r) != 0 and not self._indep_leader(r)]
        if not full_redset:
            return expand(sympy.sympify(p))
        if self._indep_leader(p):
            return expand(sympy.sympify(p))   # no jet content -> nothing to reduce
        h, r = self.R.differential_prem(p, full_redset, mode)
        return r

    def _do_reduce(self, q, A, wall_s, wall_rss_mb):
        """expand(diffprem(q, A)), optionally under a per-call fork wall.

        With no wall set this is byte-identical to the inline call.  With a
        wall, the reduction runs in a forked child (inputs ride copy-on-write,
        so nothing is serialized in); the parent caps wall-clock and child RSS,
        SIGKILLs on breach, and only an under-wall result is pickled back.  The
        swollen object, if it forms, dies in the child and never crosses the
        process boundary."""
        if not (wall_s or wall_rss_mb):
            return expand(self.diffprem(q, A))
        return self._reduce_walled(q, A, wall_s, wall_rss_mb)

    def _reduce_walled(self, q, A, wall_s, wall_rss_mb, poll=0.5):
        import os as _os, select as _select, pickle as _pickle
        import time as _time, signal as _signal
        r_fd, w_fd = _os.pipe()
        t0 = _time.time()
        pid = _os.fork()
        if pid == 0:                                   # ---- child ----
            try:
                _os.close(r_fd)
                try:
                    res = expand(self.diffprem(q, A))
                    data = _pickle.dumps(('ok', res),
                                         protocol=_pickle.HIGHEST_PROTOCOL)
                except BaseException as ex:            # noqa: BLE001
                    data = _pickle.dumps(('err', repr(ex)))
                mv = memoryview(data)
                while mv:
                    n = _os.write(w_fd, mv[:1 << 20])
                    mv = mv[n:]
            finally:
                _os._exit(0)                           # no atexit / no flush
        # ---- parent ----
        _os.close(w_fd)
        buf = bytearray()
        reason = None
        try:
            while True:
                el = _time.time() - t0
                if wall_s and el > wall_s:
                    reason = 'time>%gs' % wall_s
                    break
                if wall_rss_mb:
                    rss = _child_rss_mb(pid)
                    if rss is not None and rss > wall_rss_mb:
                        reason = 'rss>%dMB(peak~%dMB)' % (wall_rss_mb, rss)
                        break
                rl, _, _ = _select.select([r_fd], [], [], poll)
                if r_fd in rl:
                    chunk = _os.read(r_fd, 1 << 20)
                    if not chunk:
                        break                          # EOF: child done
                    buf += chunk
        finally:
            if reason:
                try:
                    _os.kill(pid, _signal.SIGKILL)
                except ProcessLookupError:
                    pass
            _os.close(r_fd)
            _os.waitpid(pid, 0)
        if reason:
            raise _WalledReduction({'why': reason, 'elapsed': _time.time() - t0})
        if not buf:
            raise RuntimeError("walled child produced no output")
        tag, val = _pickle.loads(bytes(buf))
        if tag == 'err':
            raise RuntimeError("diffprem failed in walled child: %s" % val)
        return val

    def _record_walled(self, walled, walled_file, clbl, kind, leadlbl,
                       nA, q_in, info, t0):
        import time as _time, os as _os
        try:
            nin = self._nterms(q_in)
        except Exception:
            nin = -1
        rec = {'cell_label': clbl, 'kind': kind, 'leader': leadlbl,
               'nA': nA, 'in_terms': nin,
               'why': info['why'], 'elapsed': info['elapsed']}
        walled.append(rec)
        if walled_file:
            try:
                with open(walled_file, 'a') as _wf:
                    _wf.write("walled %d | v#%s | %s ld=%s | |A|=%d | "
                              "in=%d terms | %s | t=%.1fs\n" % (
                                  len(walled), clbl, kind, leadlbl, nA, nin,
                                  info['why'], _time.time() - t0))
                    _wf.flush(); _os.fsync(_wf.fileno())
            except Exception:
                pass

    def nf(self, f, Nlist):
        """Groebner[NormalForm](f, N, plex(parms))."""
        if not Nlist:
            return expand(sympy.sympify(f))
        return self.PR.normal_form(f, Nlist)

    # -----------------------------------------------------------------
    # tracing helpers (PRG_TRACE) -- label & pretty-print worklist vertices
    # -----------------------------------------------------------------
    def _nterms(self, e):
        """Number of monomials in e (a crude swell metric)."""
        try:
            return len(sympy.expand(sympy.sympify(e)).as_ordered_terms())
        except Exception:
            return -1

    def _lds(self, p):
        """Leading derivative of p as a compact string (or 'const')."""
        try:
            if is_constant(p):
                return 'const'
            return str(self.ld(p))
        except Exception:
            return '?'

    def _short(self, e, n=64):
        s = str(e)
        return s if len(s) <= n else '%s..(%d terms)' % (s[:n], self._nterms(e))

    def _deriv_parts(self, d):
        """(base_function, {var: order}) for a leader; order-0 leaders -> {}."""
        if isinstance(d, sympy.Derivative):
            return d.expr, {v: c for v, c in d.variable_count}
        return d, {}

    def _pair_leader(self, pair):
        """Selection rank-key for a critical pair: the lcm-derivative of its two
        leaders (the common derivative RG differentiates both equations up to).
        For same-base leaders this lifts the pair above either constituent
        (Ps_x, Ps_y -> Ps_xy), so order-2 coherence pairs correctly defer behind
        order-1 P-equations.  Different-base pairs fall back to the higher leader."""
        a = self.ld(SIM(pair[0]))
        b = self.ld(SIM(pair[1]))
        ba, ca = self._deriv_parts(a)
        bb, cb = self._deriv_parts(b)
        if ba != bb:
            return self.R.sort([a, b], 'ascending')[-1]
        counts = {v: max(ca.get(v, 0), cb.get(v, 0)) for v in set(ca) | set(cb)}
        if not counts:
            return ba
        args = []
        for v, c in counts.items():
            args += [v] * c
        return sympy.Derivative(ba, *args)

    def _select_lowest(self, V):
        """Lowest-leader-first selection across pending P-equations and D-pairs
        (BLAD bad_pick_and_remove_quadruple_lower_leader_first).  Returns
        ('P', i) | ('D', i) | None; ties resolve P-before-D then lowest index."""
        # Round-trip each leader through R.sort so its representation matches
        # what R.sort returns below (R.sort re-canonicalises derivatives, so
        # raw srepr keys would not match the sorted output's srepr).
        norm = lambda d: self.R.sort([d], 'ascending')[0]
        Pl = [norm(self.ld(SIM(e))) for e in V[P]]
        Dl = [norm(self._pair_leader(pr)) for pr in V[D]]
        leaders = Pl + Dl
        if not leaders:
            return None
        ranked = self.R.sort(leaders, 'ascending')
        rank = {}
        for idx, d in enumerate(ranked):
            rank.setdefault(sympy.srepr(d), idx)
        cands = [(rank[sympy.srepr(d)], 0, i, 'P') for i, d in enumerate(Pl)]
        cands += [(rank[sympy.srepr(d)], 1, i, 'D') for i, d in enumerate(Dl)]
        _r, _k, i, kind = min(cands)
        return (kind, i)

    def _indep_leader_coeffs(self, q):
        """Handle an equation whose leader is an independent variable.

        Dependent-variable jets rank above the derivations, so a derivation-symbol
        leader means q carries NO jet content -- it is a polynomial in x,y,z over
        the parameter ring.  Since x,y,z are free, q=0 holds iff every coefficient
        (a parameter polynomial) vanishes.  Returns:
          None             -- q's leader is not an independent variable (skip);
          ('NP', None)     -- a coefficient is a nonzero constant => inconsistent;
          ('N', [c, ...])  -- the nonzero, non-constant parameter coefficients to
                              force to zero (pushed to P; MakeTree routes them to N).
        """
        try:
            lead = self.ld(q)
        except Exception:
            return None
        if not any(lead == d for d in self.derivations):
            return None
        coeffs = []
        for c in sympy.Poly(expand(sympy.sympify(q)), *self.derivations).coeffs():
            c = SIM(c)
            if c == 0:
                continue
            if is_constant(c):
                return ('NP', None)
            coeffs.append(c)
        return ('N', coeffs)

    def _fmt_entry(self, v, parent, rule, full):
        """Render a sextuplet when it is entered into the tree.

        A and P are shown by their *leaders* (the chain structure); the
        parametric/inequation sets (Sineq, S2, N, W) are shown as short polys.
        With PRG_TRACE=full every component is printed in full.
        """
        par = ('v#%s' % parent) if parent is not None else '(root)'
        lines = ["  + v#%s  <- %s  [%s]" % (v[LBL], par, rule)]

        def row(name, lst, mode):
            if not lst:
                return "       %-7s: -" % name
            if full:
                body = " ;  ".join(str(x) for x in lst)
            elif mode == 'ld':
                body = ", ".join(self._lds(x) for x in lst)
            else:
                body = ", ".join(self._short(x) for x in lst)
            return "       %-7s: %s" % (name, body)

        lines.append(row('A lead', v[A], 'ld'))
        lines.append(row('P lead', v[P], 'ld'))
        lines.append("       %-7s: %d pairs" % ('D', len(v[D])))
        lines.append(row('Sineq', v[Sineq], 'short'))
        lines.append(row('S2', v[S2], 'short'))
        lines.append(row('N', v[N], 'short'))
        lines.append(row('W', v[W], 'short'))
        return "\n".join(lines)

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
                if getattr(self, '_diff_basefield', False) and self._param_free(NC[1][0]):
                    return [v1]   # saturate parameter-free condition: drop degenerate
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
                if getattr(self, '_diff_basefield', False):
                    out = [v1]   # saturate parameter-free conditions: drop their degenerates
                    if not self._param_free(NC[1][0]):
                        out.append(v2)
                    if not self._param_free(NC[1][1]):
                        out.append(v3)
                    return out
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
    # remove_redundant  (minimal-decomposition post-pass; NOT in par-rga)
    # -----------------------------------------------------------------
    def _diff_contains(self, A_chain, C_chain):
        """True if V(C) subset V(A): every generator of the chain A reduces to
        0 modulo the regular chain C (membership in sat(C) = {C}).

        Sound but not complete -- a false 'no' (e.g. across mismatched leader
        sets, where leader-directed reduction can stall) only leaves a
        redundant component in place; it never reports a containment that does
        not hold, so it cannot drop an essential component."""
        if not C_chain:
            return all(sympy.expand(sympy.sympify(g)) == 0 for g in A_chain)
        for g in A_chain:
            try:
                if sympy.expand(self.diffprem(g, C_chain)) != 0:
                    return False
            except Exception:
                return False
        return True

    def _cell_empty(self, Z, NZ):
        """Is the constructible cell  V(Z) \\ V(prod NZ)  empty?

        Empty iff prod(NZ) vanishes on all of V(Z), i.e. prod(NZ) in
        radical(<Z>).  With no zero-conditions the cell is a complement of a
        hypersurface (non-empty); with prod(NZ)==0 it is empty."""
        prod = sympy.Integer(1)
        for w in NZ:
            prod = sympy.expand(prod * sympy.sympify(w))
        if sympy.sympify(prod) == 0:
            return True
        if not Z:
            return False
        return self.PR.radical_membership(prod, list(Z)) is True

    def _cell_covered(self, cellC, container_cells):
        """Is cell(C)=(N_C, W_C) covered by the union of container_cells (each
        an (N, W) pair)?  Computes  cell(C) minus the union  as a list of
        constructible (zeros, nonzeros) regions -- subtracting a cell (Na=0,
        Wa!=0) splits a region into {n!=0 for n in Na} and {w==0 for w in Wa}
        -- and reports True iff every surviving region is empty."""
        Nc, Wc = cellC
        regions = [(list(Nc), list(Wc))]
        for (Na, Wa) in container_cells:
            nxt = []
            for (Z, NZ) in regions:
                for n in Na:
                    nxt.append((Z, NZ + [n]))        # region & {n != 0}
                for w in Wa:
                    nxt.append((Z + [w], NZ))        # region & {w == 0}
            regions = [(Z, NZ) for (Z, NZ) in nxt if not self._cell_empty(Z, NZ)]
            if not regions:
                return True
        return len(regions) == 0

    def remove_redundant(self, systems):
        """OPT-IN, EXPERIMENTAL.  Drop components whose solution set is covered
        by the union of the others.  NOT part of Fakouri's algorithm (par-rga
        emits every consistent regular representation, redundant ones included,
        e.g. the initial-degenerate v=0 branch of a non-monic equation), so it
        is off by default; MainProc(..., minimal=True) enables it.

        A component C is dropped if the other components that differentially
        contain it (V(C) subset V(A_i), tested by reducing A_i's generators
        modulo C's chain) already cover cell(C) in parameter space.  Removed one
        at a time so duplicates are not both dropped.

        CAVEATS -- this does NOT reproduce the paper's tables.
          * It over-collapses: where a characterizable component is reducible
            (V = V(P1) u V(P2)), it keeps only the coarse cover and drops the
            finer pieces the paper lists separately.  Verified sound on P1/P3
            (the kept union still covers stock RosenfeldGroebner's components)
            but coarser than the published decomposition.
          * Soundness relies on V(C) subset V(A_i) holding on all of
            cell(C) ∩ cell(A_i); the diffprem test reduces over Q(params) and
            could in principle report a containment that holds only generically.
            Not certified at special parameter values."""
        systems = list(systems)
        changed = True
        while changed:
            changed = False
            for idx in range(len(systems)):
                C = systems[idx]
                C_chain = [p for p in C[0] if sympy.sympify(p) != 0]
                container_cells = []
                for j, Ao in enumerate(systems):
                    if j == idx:
                        continue
                    A_chain = [p for p in Ao[0] if sympy.sympify(p) != 0]
                    if self._diff_contains(A_chain, C_chain):
                        container_cells.append((list(Ao[2]), list(Ao[3])))
                if container_cells and self._cell_covered(
                        (list(C[2]), list(C[3])), container_cells):
                    del systems[idx]
                    changed = True
                    break
        return systems

    # -----------------------------------------------------------------
    # MainProc  (Algorithm 4.1)
    # -----------------------------------------------------------------
    def MainProc(self, Pin, Qin=None, max_vertices=20000, wall_timeout=None,
                 progress_every=0, minimal=False,
                 diffprem_wall_s=None, diffprem_wall_rss_mb=None,
                 walled_file=None):
        if Qin is None:
            Qin = []
        """Run the parametric RG on equations Pin (=0) and inequations Qin (!=0).

        Returns the list of consistent regular representations, each as
        [A_list, S_list, N_list, W_list].

        ``wall_timeout`` (seconds) and ``max_vertices`` bound the search; on
        either limit we raise RuntimeError with the partial Decom attached to
        the exception (``.partial``) so callers can record a time-boxed result.
        """
        import time as _time, os as _os, sys as _sys
        _tr = _os.environ.get('PRG_TRACE', '')
        trace = bool(_tr)
        full = (_tr.lower() == 'full')
        _dump_at = _os.environ.get('PRG_DUMP_AT')   # label whose critical pair
                                                    # (and chain) to dump in full
        # PRG_PEQ_FIRST: mimic BLAD's lower-leader-first selection by draining
        # pending P-equations before discharging critical pairs.  In this system
        # every chain-rule coherence pair has an order-2 lcm-leader (e.g. Ps_xy)
        # that outranks the order-<=1 DPs P-equations resolving it, so P-first
        # coincides with lower-leader-first on exactly those pairs: the pair then
        # reduces to 0 against the completed chain instead of forcing a split on
        # its initial (the spurious Vf_y-style branches).  Confluence-safe (RG's
        # decomposition is selection-order-independent for a fixed ranking).
        _peq_first = bool(_os.environ.get('PRG_PEQ_FIRST'))
        # PRG_LOWER_LEADER: true BLAD-style lowest-leader-first selection across
        # both P-equations and critical pairs (keyed on the lcm-derivative for
        # pairs).  Unlike PRG_PEQ_FIRST (drain-all-P-then-D, which front-loads the
        # full chain), this discharges low-order coherence pairs while the chain
        # is still small, before high-leader equations (the ODE) enter.
        _lower_leader = bool(_os.environ.get('PRG_LOWER_LEADER'))
        # PRG_DIFF_BASEFIELD: saturate parameter-free initials/separants (drop
        # their degenerate =0 branch) like BLAD's base field, splitting only on
        # parameter-bearing conditions.  Makes the generic branch track BLAD's
        # single component, localising the divergence to the parametric splits.
        self._diff_basefield = bool(_os.environ.get('PRG_DIFF_BASEFIELD'))
        # PRG_DECOM_FILE: append each terminal cell's (N,W) to this file as it is
        # found, so a SIGKILL (OOM) on a long run does not lose the results.
        _decom_file = _os.environ.get('PRG_DECOM_FILE')
        self._lblctr = 0

        def _emit(s):
            print(s)
            _sys.stdout.flush()

        def _enq(vlist, parent, rule):
            """Assign a fresh label to each vertex as it enters the tree, print
            its sextuplet (when tracing), and return the labelled list."""
            out = []
            for v in vlist:
                v = list(v)
                if len(v) <= LBL:
                    v += [None] * (LBL + 1 - len(v))
                self._lblctr += 1
                v[LBL] = self._lblctr
                if trace:
                    _emit(self._fmt_entry(v, parent, rule, full))
                out.append(v)
            return out

        Decom = []
        NP = []
        walled = []
        self.walled = walled
        root = [[], list(Pin), [], list(Qin), [], [], []]
        Br = _enq([root], None, 'root')
        count = 0
        t0 = _time.time()
        while Br:
            count += 1
            if count > max_vertices:
                e = RuntimeError("vertex budget exceeded (%d)" % max_vertices)
                e.partial = (Decom, NP, len(Br), count)
                raise e
            if wall_timeout is not None and (_time.time() - t0) > wall_timeout:
                e = RuntimeError("wall timeout (%ss) after %d vertices"
                                 % (wall_timeout, count))
                e.partial = (Decom, NP, len(Br), count)
                raise e
            if progress_every and count % progress_every == 0:
                print("  [prg] vertex %d  |Br|=%d  |Decom|=%d  |NP|=%d  %.0fs"
                      % (count, len(Br), len(Decom), len(NP), _time.time() - t0))
                _sys.stdout.flush()
            cur = Br[0]
            Br = Br[1:]
            clbl = cur[LBL] if len(cur) > LBL else '?'
            _vt = _time.time()
            T = self.CheckBranch(cur)
            if T is False:
                if trace:
                    _emit(">> v#%s  => NP (CheckBranch: inconsistent)  %.1fs  t=%.1fs"
                          % (clbl, _time.time() - _vt, _time.time() - t0))
                NP.append(cur)
                continue
            V = T
            if trace:
                _emit(">> v#%s  processing  |A|=%d |P|=%d |D|=%d |N|=%d |W|=%d  t=%.1fs"
                      % (clbl, len(V[A]), len(V[P]), len(V[D]),
                         len(V[N]), len(V[W]), _time.time() - t0))
            # ---- choose the next work item ----
            if _lower_leader and (V[D] or V[P]):
                sel = self._select_lowest(V)
            elif V[D] and not (_peq_first and V[P]):
                sel = ('D', 0)
            elif V[P]:
                sel = ('P', 0)
            else:
                sel = None

            if sel is not None and sel[0] == 'D':
                # solve a critical pair via its DeltaPolynomial
                pair = V[D][sel[1]]
                if _dump_at is not None and str(clbl) == _dump_at:
                    _emit("==== DUMP critical pair at v#%s  (chain |A|=%d) ====" % (clbl, len(V[A])))
                    _emit("  pair[0] (ld=%s) = %s" % (self._lds(pair[0]), sympy.sympify(pair[0])))
                    _emit("  pair[1] (ld=%s) = %s" % (self._lds(pair[1]), sympy.sympify(pair[1])))
                    for _i, _a in enumerate(V[A]):
                        _emit("  A[%d] (ld=%s) = %s" % (_i, self._lds(_a), sympy.sympify(_a)))
                    _sys.stdout.flush()
                _td = _time.time()
                q = DeltaPolynomial(self.R, pair[0], pair[1], self.derivations)
                _n0, _t0d = self._nterms(q), _time.time() - _td
                if _dump_at is not None and str(clbl) == _dump_at:
                    _emit("  >> Δ computed: %d terms in %.1fs; now reducing by chain (diffprem)..." % (_n0, _t0d))
                    _sys.stdout.flush()
                # par-rga reduces q by the parameter-derivatives first; those are
                # 0 on the stock binding (parameters are constants), so skip.
                _tp = _time.time()
                try:
                    q = self._do_reduce(q, V[A], diffprem_wall_s,
                                        diffprem_wall_rss_mb)
                except _WalledReduction as _wr:
                    self._record_walled(walled, walled_file, clbl, 'D',
                                        "%s,%s" % (self._lds(pair[0]),
                                                   self._lds(pair[1])),
                                        len(V[A]), q, _wr.info, t0)
                    if progress_every:
                        print("  [prg] vertex %d  WALLED (D-pair, %s)  %.0fs"
                              % (count, _wr.info['why'], _time.time() - t0))
                        _sys.stdout.flush()
                    continue
                _n1, _t1d = self._nterms(q), _time.time() - _tp
                if V[N]:
                    q = self.nf(q, V[N])
                q = SIM(q)
                if trace:
                    _emit("     D-pair Δ(%s, %s): Δ=%d terms (%.1fs)  ->diffprem=%d terms (%.1fs)  ->reduced=%d terms"
                          % (self._lds(pair[0]), self._lds(pair[1]),
                             _n0, _t0d, _n1, _t1d, self._nterms(q)))
                if is_constant(q) and q != 0:
                    if trace: _emit("     => NP (Δ reduces to a nonzero constant)")
                    NP.append(V)
                elif q == 0:
                    nv = list(V)
                    nv[D] = Extract(V[D], [pair])
                    if trace: _emit("     => Δ=0: pair already implied; drop it, requeue")
                    Br = _enq([nv], clbl, 'Δ=0 drop-pair') + Br
                else:
                    _ilc = self._indep_leader_coeffs(q)
                    if _ilc is not None and _ilc[0] == 'NP':
                        if trace: _emit("     => NP (indep-var leader, nonzero-constant coefficient)")
                        NP.append(V)
                    elif _ilc is not None:
                        nv = list(V)
                        nv[D] = Extract(V[D], [pair])
                        nv[P] = ADD(_ilc[1], V[P])
                        if trace: _emit("     => indep-var leader (ld=%s): %d coefficient cond(s) -> N"
                                        % (self._lds(q), len(_ilc[1])))
                        Br = _enq([nv], clbl, 'indep-var coeffs') + Br
                    else:
                        q = DIVIDE(q, list(ADD(V[S2], V[Sineq])) + list(V[W]))
                        if is_constant(q):
                            if trace: _emit("     => NP (constant after DIVIDE)")
                            NP.append(V)
                        else:
                            v1 = list(V)
                            v1[D] = Extract(V[D], [pair])
                            Br4 = self.MakeTree(q, v1)
                            if trace: _emit("     => MakeTree(q, ld=%s) -> %d child(ren)"
                                            % (self._lds(q), len(Br4)))
                            Br = _enq(Br4, clbl, 'MakeTree/D') + Br
            elif sel is not None:   # 'P'
                q0 = V[P][sel[1]]
                q = q0
                _qld = self._lds(q)
                if V[N]:
                    q = self.nf(q, V[N])
                _tp = _time.time()
                try:
                    q = self._do_reduce(q, V[A], diffprem_wall_s,
                                        diffprem_wall_rss_mb)
                except _WalledReduction as _wr:
                    self._record_walled(walled, walled_file, clbl, 'P',
                                        _qld, len(V[A]), q, _wr.info, t0)
                    if progress_every:
                        print("  [prg] vertex %d  WALLED (P-eqn, %s)  %.0fs"
                              % (count, _wr.info['why'], _time.time() - t0))
                        _sys.stdout.flush()
                    continue
                _t1d = _time.time() - _tp
                q = SIM(q)
                if trace:
                    _emit("     P-eqn (ld=%s) ->diffprem=%d terms (%.1fs)"
                          % (_qld, self._nterms(q), _t1d))
                if is_constant(q) and q != 0:
                    if trace: _emit("     => NP (eqn reduces to a nonzero constant)")
                    NP.append(V)
                elif q == 0:
                    nv = list(V)
                    nv[P] = Extract(V[P], [q0])
                    if trace: _emit("     => eqn=0: already implied; drop it, requeue")
                    Br = _enq([nv], clbl, 'eqn=0 drop') + Br
                else:
                    _ilc = self._indep_leader_coeffs(q)
                    if _ilc is not None and _ilc[0] == 'NP':
                        if trace: _emit("     => NP (indep-var leader, nonzero-constant coefficient)")
                        NP.append(V)
                    elif _ilc is not None:
                        nv = list(V)
                        nv[P] = ADD(_ilc[1], Extract(V[P], [q0]))
                        if trace: _emit("     => indep-var leader (ld=%s): %d coefficient cond(s) -> N"
                                        % (self._lds(q), len(_ilc[1])))
                        Br = _enq([nv], clbl, 'indep-var coeffs') + Br
                    else:
                        q = DIVIDE(q, list(ADD(V[S2], V[Sineq])) + list(V[W]))
                        if is_constant(q):
                            if trace: _emit("     => NP (constant after DIVIDE)")
                            NP.append(V)
                        else:
                            v1 = list(V)
                            v1[P] = Extract(V[P], [q0])
                            Br4 = self.MakeTree(q, v1)
                            if trace: _emit("     => MakeTree(q, ld=%s) -> %d child(ren)"
                                            % (self._lds(q), len(Br4)))
                            Br = _enq(Br4, clbl, 'MakeTree/P') + Br
            else:
                if trace: _emit("     => Decom (terminal regular system)  ✓")
                Decom.append(V)
                if _decom_file:
                    # durable per-cell checkpoint (survives an OOM SIGKILL)
                    try:
                        with open(_decom_file, 'a') as _df:
                            _df.write("cell %d | v#%s | t=%.1fs | |A|=%d | N=%s | W=%s\n" % (
                                len(Decom), clbl, _time.time() - t0, len(V[A]),
                                sorted(str(expand(sympy.sympify(n))) for n in V[N]),
                                sorted(str(expand(sympy.sympify(w))) for w in V[W])))
                            _df.flush(); _os.fsync(_df.fileno())
                    except Exception:
                        pass
        # assemble [A, S+S2, N, W] for each terminal vertex
        systems = [[V[A], ADD(V[Sineq], V[S2]), V[N], V[W]] for V in Decom]
        de = self.checkregular(systems)
        if minimal:
            de = self.remove_redundant(de)
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
