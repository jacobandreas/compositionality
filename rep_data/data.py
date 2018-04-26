from collections import namedtuple
import numpy as np
import torch
from torch.autograd import Variable

Batch = namedtuple('Batch', ('tgt', 'ctx'))
CompBatch = namedtuple('CompBatch', ('bi', 'uni1', 'uni2'))

class Dataset(object):
    def __init__(self):
        vocab = {}
        rev_vocab = {}
        with open('data/vocab.txt') as fh:
            for line in fh:
                word, id = line.rsplit(',', 1)
                id = int(id)
                vocab[word] = id
                rev_vocab[id] = word
            self._vocab = vocab
            self._rev_vocab = rev_vocab

        uni_tgt = np.load('data/uni_tgt.npy')
        uni_ctx = np.load('data/uni_ctx.npy')
        bi_tgt = np.load('data/bi_tgt.npy')
        bi_ctx = np.load('data/bi_ctx.npy')
        self._tgt = np.concatenate((uni_tgt, bi_tgt))
        self._ctx = np.concatenate((uni_ctx, bi_ctx))

        bis = []
        uni1s = []
        uni2s = []
        with open('data/bigrams.txt') as fh:
            for line in fh:
                bi, uni1, uni2 = (int(i) for i in line.split(','))
                bis.append(bi)
                uni1s.append(uni1)
                uni2s.append(uni2)
        self._comp_batch = CompBatch(bis, uni1s, uni2s)

        self.n_vocab = len(vocab)
        self.n_examples = self._tgt.shape[0]

    def get_batch(self, size):
        ids = np.random.randint(self.n_examples, size=size)
        tgt = self._tgt[ids]
        ctx = self._ctx[ids, :]
        return Batch(
            Variable(torch.LongTensor(tgt)),
            Variable(torch.LongTensor(ctx)))

    def get_comp_batch(self):
        return self._comp_batch

    def unencode(self, i):
        return self._rev_vocab[i]
