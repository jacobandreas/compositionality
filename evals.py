import numpy as np
from scipy import stats

def metric(es, fs, sim_e, sim_f):
    sims = []
    pairs = list(zip(es, fs))
    random = np.random.RandomState(0)
    random.shuffle(pairs)
    for e1, f1 in pairs[:300]:
        for e2, f2 in pairs[-300:]:
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
