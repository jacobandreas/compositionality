from collections import namedtuple
import numpy as np
import torch
from torch.autograd import Variable

MSet = namedtuple('MSet', ('id', 'lf', 'labels'))
Batch = namedtuple('Batch',
    ('mtrain_feats', 'mtrain_labels', 'mpred_feats', 'mpred_labels',
        'indices', 'mids', 'lfs'))

class Dataset(object):
    def __init__(self):
        vocab = {}
        with open('data/vocab.txt') as fh:
            for line in fh:
                word, id = line.split(',')
                vocab[word] = id
        self._vocab = vocab

        things = []
        with open('data/things.txt') as fh:
            for line in fh:
                _, feat_ids = line.split(',')
                feat_ids = set(int(i) for i in feat_ids.split())
                feats = [1 if i in feat_ids else 0 for i in range(len(vocab))]
                things.append(feats)
        self._things = things

        msets = []
        with open('data/labels.txt') as fh:
            for line in fh:
                id, lf, labels = line.split(',')
                labels = [int(i) for i in labels.split()]
                msets.append(MSet(id, lf, labels))
        self._msets = msets

        with open('data/train_ids.txt') as fh:
            self._train_ids = [int(i) for i in fh]
        with open('data/val_ids.txt') as fh:
            self._val_ids = [int(i) for i in fh]
        with open('data/test_ids.txt') as fh:
            self._test_ids = [int(i) for i in fh]

        self.n_things = len(things)
        self.n_features = len(vocab)

    def name(self, mid, indices, labels):
        assert len(indices) == len(labels)
        nindices = [i for i in range(len(self._things)) if i not in indices]
        nlabels = self._msets[mid].labels
        candidates = []
        for mset in self._msets:
            score1 = sum(1 for i, ii in enumerate(indices) if mset.labels[ii] != labels[i])
            score2 = sum(1 for ii in nindices if mset.labels[ii] != nlabels[ii])
            candidates.append((score1 + score2, mset.lf))
        return min(candidates)[1]

    def get_train_batch(self, size, msize):
        ids = np.random.choice(self._train_ids, size=size)
        return self._get_batch(ids, msize, fix_shuffle=False)

    def get_val_batch(self, msize):
        return self._get_batch(self._val_ids, msize)

    def get_full_batch(self, msize):
        return self._get_batch(range(len(self._msets)), msize)

    def _get_batch(self, ids, msize, fix_shuffle=True):
        rand = np.random.RandomState(0) if fix_shuffle else np.random
        size = len(ids)
        predsize = len(self._things) - msize
        mtrain_feats = np.zeros((size, msize, len(self._vocab)))
        mtrain_labels = np.zeros((size, msize, 2), dtype=np.int32)
        mpred_feats = np.zeros((size, predsize, len(self._vocab)))
        mpred_labels = np.zeros((size, predsize), dtype=np.int32)
        indices = []
        for i, i_mset in enumerate(ids):
            ii_thing = list(range(len(self._things)))
            rand.shuffle(ii_thing)
            for j, i_thing in enumerate(ii_thing[:msize]):
                mtrain_feats[i, j, :] = self._things[i_thing]
                mtrain_labels[i, j, self._msets[i_mset].labels[i_thing]] = 1
            for j, i_thing in enumerate(ii_thing[msize:]):
                mpred_feats[i, j, :] = self._things[i_thing]
                mpred_labels[i, j] = self._msets[i_mset].labels[i_thing]
            indices.append((ii_thing[:msize], ii_thing[msize:]))
        lfs = [self._msets[i_mset].lf for i_mset in ids]
        return Batch(
            Variable(torch.FloatTensor(mtrain_feats)), 
            Variable(torch.FloatTensor(mtrain_labels)), 
            Variable(torch.FloatTensor(mpred_feats)), 
            Variable(torch.FloatTensor(mpred_labels)),
            indices, ids, lfs)
