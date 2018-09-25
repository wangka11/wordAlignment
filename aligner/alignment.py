#!/usr/bin/env python
import optparse
import sys
import copy
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with EM algorithm...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


temp = {}
idx = 0
for ff, ee in bitext:
    # compute normalization
    for f in ff:
        for e in ee:
            temp[idx] = (f, e)
            idx += 1

lst = list(set(temp.values()))
pair = {}
for (i, _tuple) in enumerate(lst):
    pair[i] = _tuple

temp = []
for ff, ee in bitext:
    temp = temp + ff
temp = list(set(temp))

t = {}
# for key in pair.values():
#     t[key] = 0.25  # initialize t(e|f) uniformly

for ff, ee in bitext:
    for f in ff:
        for e in ee:
            t[(f, e)] = float(1) / len(temp)


def checkConvergence(d1, d2):
    if (not d1 or not d2): return False
    tol = 0.001
    equals = lambda a, b: abs(a - b) < tol
    convergences = [equals(d1[key], d2[key]) for key in set(d1.keys()) & set(d2.keys())]
    num = convergences.count(True)
    return float(num) / len(convergences) >= 0.90

K = 0
old = {}
while True:  # while not converged

    if (checkConvergence(old, t)): break
    old = copy.copy(t)
    count, total = {}, {}
    for key in pair.values():
        count[key] = 0
    for _tuple in pair.values():
        total[_tuple[1]] = 0
    s_total = {}
    for ff, ee in bitext:
        # compute normalization
        for f in ff:
            s_total[f] = 0
            for e in ee:
                s_total[f] += t[(f, e)]
        # collect counts
        for f in ff:
            for e in ee:
                count[(f, e)] += t[(f, e)] / s_total[f]
                total[e] += t[(f, e)] / s_total[f]
    # estimate probabilities
    for f, e in pair.values():
        t[(f, e)] = count[(f, e)] / total[e]
    # end of while
    K += 1


for (f, e) in bitext:
    for (i, f_i) in enumerate(f):
        max_p = 0.0
        best_j = 0
        for (j, e_j) in enumerate(e):
            if t[(f_i, e_j)] > max_p:
                max_p = t[(f_i, e_j)]
                best_j = j
        sys.stdout.write("%i-%i " % (i,best_j))
    sys.stdout.write("\n")