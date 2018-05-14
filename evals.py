from collections import defaultdict
import editdistance
import numpy as np
from scipy import stats
import sexpdata
import zss

def l1_dist(x, y):
    return np.abs(x-y).sum()

def l2_dist(x, y):
    return ((x-y)**2).sum()

def cos_dist(x, y):
    return 1 - ((x/np.linalg.norm(x) * y/np.linalg.norm(y))).sum()

_tree_dist_cache = {}
def tree_dist(x, y):
    key = (x, y)
    if key not in _tree_dist_cache:
        x = sexpdata.loads(x)
        y = sexpdata.loads(y)
        dist = zss.simple_distance(
            x,
            y,
            lambda w: w[1:] if isinstance(w, list) else [],
            lambda w: w[0].value() if isinstance(w, list) else w.value(),
            lambda w, z: 0 if w == z else 1)
        _tree_dist_cache[key] = dist
    return _tree_dist_cache[key]

_str_dist_cache = {}
def str_dist(x, y):
    key = (x, y)
    if key not in _str_dist_cache:
        dist = editdistance.eval(x, y)
        _str_dist_cache[key] = dist
    return _str_dist_cache[key]

def my_tree_size(x):
    if isinstance(x, tuple):
        x1, x2 = x
        return my_tree_size(x1) + my_tree_size(x2)
    return 1

_my_tree_dist_cache = {}
def my_tree_dist(x, y):
    if x == y:
        return 0
    if (x, y) not in _my_tree_dist_cache:
        opts = []
        if isinstance(x, tuple):
            x1, x2 = x
            opts.append(my_tree_dist(x1, y) + my_tree_size(x2))
            opts.append(my_tree_dist(X2, y) + my_tree_Size(x1))
        if isinstance(y, tuple):
            y1, y2 = y
            opts.append(my_tree_dist(x, y1) + my_tree_size(y2))
            opts.append(my_tree_dist(x, y2) + my_tree_size(y1))
        if isinstance(x, tuple) and isinstance(y, tuple):
            my_tree_dist(x1, y1) + my_tree_dist(x2, y2),
            my_tree_dist(x1, y2) + my_tree_dist(x2, y1),
        _my_tree_dist_cache[x, y] = min(opts)
    return _my_tree_dist_cache[x, y]


def comp_eval(r_prim, e_prim, r, e, r_comp, r_dist):
    prim_cache = defaultdict(list)
    for rp, ep in zip(r_prim, e_prim):
        prim_cache[ep].append(rp)
    prim_cache = {k: np.mean(v, axis=0) for k, v in prim_cache.items()}

    def reconstruct(e):
        if isinstance(e, tuple):
            e1, e2 = e
            return r_comp(reconstruct(e1), reconstruct(e2))
        return prim_cache[e]

    err = []
    for rep, exp in zip(r, e):
        reconst_rep = reconstruct(exp)
        err.append(r_dist(rep, reconst_rep))
    return err

def isom(es, fs, sim_e, sim_f, size=100):
    sims = []
    pairs = list(zip(es, fs))
    random = np.random.RandomState(0)
    random.shuffle(pairs)
    for e1, f1 in pairs[:size]:
        for e2, f2 in pairs[-size:]:
            sims.append((sim_e(e1, e2), sim_f(f1, f2)))
    e_sims, f_sims = zip(*sims)
    r = stats.spearmanr(e_sims, f_sims)
    return r.correlation

def hom(preds, trues, sim, regress=False):
    if regress:
        assert False
        op, residuals, rank, _ = np.linalg.lstsq(preds, trues)
        return residuals.mean()
    else:
        return np.mean([sim(p, t) for p, t in zip(preds, trues)])
