#!/usr/bin/env python3

from collections import defaultdict
import itertools as it
import numpy as np

np.random.seed(0)

ATTRIBUTES = [
    ['small', 'medium', 'large'],
    ['red', 'orange', 'yellow', 'green', 'blue'],
    ['circle', 'square', 'triangle', 'pentagon'],
    #['solid', 'dashed', 'dotted'],
    ['top', 'middle', 'bottom', 'left', 'right']
]

LITERALS = ['true', 'false']

#UNARIES = ['not']
UNARIES = []

NARIES = ['and', 'or']
#BINARIES = []

MAX_DEPTH = 2

def things():
    for group in it.product(*ATTRIBUTES):
        yield group

def lfs(depth):
    if depth == 1:
        for attr_group in ATTRIBUTES:
            for attr in attr_group:
                yield attr
        for literal in LITERALS:
            yield literal
    else:
        for unary in UNARIES:
            for d in range(1, depth):
                for lf in lfs(d):
                    yield (unary, lf)
        #for binary in BINARIES:
        for nary in NARIES:
            available_depths = range(1, depth)
            for depths in it.chain(
                    it.product(available_depths, repeat=2),
                    it.product(available_depths, repeat=3)):
                depths = list(depths)
                gen_lfs = [lfs(d) for d in depths]
                for lf_group in it.product(*gen_lfs):
                    yield (nary,) + lf_group

def evaluate(lf, thing):
    if isinstance(lf, tuple):
        fn = lf[0]
        if fn == 'not':
            return not evaluate(lf[1], thing)
        elif fn == 'and':
            return all(evaluate(l, thing) for l in lf[1:])
        elif fn == 'or':
            return any(evaluate(l, thing) for l in lf[1:])
        else:
            return False
    elif lf == 'true':
        return True
    elif lf == 'false':
        return False
    else:
        return lf in thing

def size(lf):
    if isinstance(lf, tuple):
        return 1 + sum(size(l) for l in lf[1:])
    else:
        return 1

def pp(lf):
    if isinstance(lf, tuple):
        return '(' + ' '.join(pp(l) for l in lf) + ')'
    else:
        return lf

base_lfs = it.chain(*(lfs(d) for d in range(1, MAX_DEPTH+1)))
all_lfs = it.chain(*([lf, ('not', lf)] for lf in base_lfs))

things = list(things())
groups = defaultdict(list)
#for depth in range(1, MAX_DEPTH+1):
#    for lf in lfs(depth):
#        sig = tuple(int(evaluate(lf, thing)) for thing in things)
#        groups[sig].append(lf)
for lf in all_lfs:
    sig = tuple(int(evaluate(lf, thing)) for thing in things)
    groups[sig].append(lf)

labeled = {k: min(v, key=size) for k, v in groups.items()}
labeled = {k: labeled[k] for k in sorted(labeled.keys(), key=sum)}

attr_vocab = {}
for attr_group in ATTRIBUTES:
    for attr in attr_group:
        attr_vocab[attr] = len(attr_vocab)

with open('things.txt', 'w') as fh:
    for thing in things:
        print('%s,%s' % (pp(thing), ' '.join(str(attr_vocab[a]) for a in thing)), file=fh)

with open('labels.txt', 'w') as fh:
    for i, (k, v) in enumerate(labeled.items()):
        print('%s,%s,%s' % (i, pp(v), ' '.join(str(kk) for kk in k)), file=fh)

with open('vocab.txt', 'w') as fh:
    for pair in attr_vocab.items():
        print('%s,%s' % pair, file=fh)

ids = list(range(len(labeled)))
np.random.shuffle(ids)
#train_ids = ids[:2450]
#val_ids = ids[2450:2700]
#test_ids = ids[2700:]
train_ids = ids[:1500]
val_ids = ids[1500:2000]
test_ids = ids[2000:]

for name, ids in [
        ('train_ids.txt', train_ids), ('val_ids.txt', val_ids),
        ('test_ids.txt', test_ids)]:
    with open(name, 'w') as fh:
        for i in ids:
            print(i, file=fh)
