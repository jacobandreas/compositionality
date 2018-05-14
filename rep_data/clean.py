#!/usr/bin/env python3

from collections import Counter
import numpy as np

FILES = ['train_raw.txt']
SEP = '__'
DROP_STOP = 50
KEEP_CONTENT = 2000

def lines(fname):
    with open(fname) as fh:
        for line in fh:
            words = line.strip().lower().split()
            words = ['##' if any(c.isdigit() for c in word) else word for word
                    in words]
            yield words

stop_counter = Counter()
for fname in FILES:
    for line in lines(fname):
        for i in range(len(line)):
            stop_counter[line[i]] += 1
stopwords = set(w for w, c in stop_counter.most_common(DROP_STOP))

counter = Counter()
for fname in FILES:
    for line in lines(fname):
        for i in range(len(line)):
            if line[i] in stopwords:
                continue
            counter[line[i]] += 1
            if i < len(line) - 1:
                if line[i+1] in stopwords:
                    continue
                bigram = line[i] + SEP + line[i+1]
                counter[bigram] += 1

vocab = {'****': 0}
counts = counter.most_common(KEEP_CONTENT)
for word, count in counts:
    vocab[word] = len(vocab)

def make_dataset(ctx):
    uni_tgt = []
    uni_ctx = []
    bi_tgt = []
    bi_ctx = []
    for fname in FILES:
        for line in lines(fname):
            for i in range(len(line)):
                unigram = line[i]
                if unigram not in vocab:
                    continue
                neighborhood = line[i-ctx:i] + line[i+1:i+ctx+1]
                neighborhood = [n for n in neighborhood if n in vocab]
                uni_tgt.append(vocab[unigram])
                uni_ctx.append([vocab[n] for n in neighborhood] + [0] * (2*ctx-len(neighborhood)))

                if i == len(line) - 1:
                    continue
                bigram = line[i] + SEP + line[i+1]
                if bigram not in vocab:
                    continue
                neighborhood = line[i-ctx:i] + line[i+2:i+ctx+2]
                neighborhood = [n for n in neighborhood if n in vocab]
                bi_tgt.append(vocab[bigram])
                bi_ctx.append([vocab[n] for n in neighborhood] + [0] * (2*ctx-len(neighborhood)))

    prefix = 'ctx_%d/' % ctx

    uni_tgt = np.asarray(uni_tgt)
    uni_ctx = np.asarray(uni_ctx)
    bi_tgt = np.asarray(bi_tgt)
    bi_ctx = np.asarray(bi_ctx)
    np.save(prefix + 'uni_tgt', uni_tgt)
    np.save(prefix + 'uni_ctx', uni_ctx)
    np.save(prefix + 'bi_tgt', bi_tgt)
    np.save(prefix + 'bi_ctx', bi_ctx)

    with open(prefix + 'vocab.txt', 'w') as fh:
        for k, v in vocab.items():
            print('%s,%s' % (k, v), file=fh)

    with open(prefix + 'bigrams.txt', 'w') as fh:
        for k in vocab:
            if SEP not in k:
                continue
            w1, w2 = k.split(SEP)
            assert w1 in vocab and w2 in vocab
            print('%s,%s,%s' % (vocab[k], vocab[w1], vocab[w2]), file=fh)

for i in [1, 3, 5, 7]:
    make_dataset(i)
