"""Ring construction + jet-notation helpers.

Translate the Maple ring definitions of par-rga.txt (F, F1) and the Maple
jet notation (u[x], u[x,x], v[x], ...) to the Python DifferentialAlgebra
API (Derivative(u(x,y,z), x), ...).
"""

import sympy
from sympy import Derivative, var
import DifferentialAlgebra as DA


def build_ring(derivation_names, blocks, parameter_names):
    """Build (R, derivations, dep_funcs, parms, jet) for a parametric ring.

    ``derivation_names``  -- e.g. ['x','y','z']
    ``blocks``            -- list of blocks, each a list of dependent-variable
                             names, highest-ranked block first; e.g.
                             [['m','u','v','w']] for the orderly single block,
                             or [['w'],['v'],['u'],['m']] for elimination.
    ``parameter_names``   -- e.g. ['a','b','c','d']

    Returns a dict-like namespace with:
        R            -- the DifferentialRing
        derivations  -- list of sympy independent-variable symbols
        dep          -- {name: sympy.Function}
        parms        -- list of sympy parameter Symbols
        jet(name, *idx) -- build a derivative u[idx...]
    """
    derivations = [var(n) for n in derivation_names]
    parms = [var(n) for n in parameter_names]
    dep = {}
    block_objs = []
    for blk in blocks:
        objs = []
        for nm in blk:
            f = sympy.Function(nm)
            dep[nm] = f
            objs.append(f)
        if len(objs) == 1:
            block_objs.append(objs[0])
        else:
            block_objs.append(objs)
    # parameters appear as a final block of 0-ary symbols
    param_block = list(parms)
    R = DA.DifferentialRing(derivations=derivations,
                            blocks=block_objs + [param_block],
                            parameters=parms)

    derivset = derivations

    def jet(name, *idx):
        """u[x], u[x,x] -> Derivative(u(x,y,z), x), Derivative(u(...), x, x)."""
        f = dep[name](*derivset)
        if not idx:
            return f
        diffs = []
        for token in idx:
            diffs.append(var(token) if isinstance(token, str) else token)
        return Derivative(f, *diffs)

    return {
        'R': R,
        'derivations': derivations,
        'dep': dep,
        'parms': parms,
        'jet': jet,
    }
