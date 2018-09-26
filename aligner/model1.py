#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import time

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-k", "--iteration", dest="iterations", default=10, help="Number of iterations (default=10)")
optparser.add_option("-t", "--trainDirection", dest="trainDirection", default="e2f", help="Translation direction (default=e2f): e2f or f2e")
# optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with EM algorithm...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

start = time.time()

def model1_train_e2f(bitext, opts):

    t = defaultdict(float)

    # Initialize t
    for ff, ee in bitext:
        for f in ff:
            for e in ee:
                t[(f, e)] = float(1) / len(ee)

    k = 0

    iterations = int(opts.iterations)
    while k < iterations:
        k += 1
        # Initilaize all the counts
        count_e = defaultdict(float)
        count_fe = defaultdict(float)

        # E step: compute expected counts
        for ff, ee in bitext:
            for f in set(ff):
                Z = 0
                for e in set(ee):
                    Z += t[(f, e)]

                for e in set(ee):
                    c = float(t[(f, e)]) / Z
                    count_fe[(f, e)] += c
                    count_e[e] += c

        #  M step: normalize
        for (f, e) in count_fe.keys():
            t[(f, e)] = count_fe[(f, e)] / count_e[e]

    return t

def model1_train_f2e(bitext, opts):

    t = defaultdict(float)

    # Initialize t
    for ff, ee in bitext:
        for e in ee:
            for f in ff:
                t[(e, f)] = float(1) / len(ff)

    k = 0

    iterations = int(opts.iterations)
    while k < iterations:
        k += 1
        # Initilaize all the counts
        count_f = defaultdict(float)
        count_ef = defaultdict(float)

        # E step: compute expected counts
        for ff, ee in bitext:
            for e in set(ee):
                Z = 0
                for f in set(ff):
                    Z += t[(e, f)]

                for f in set(ff):
                    c = float(t[(e, f)]) / Z
                    count_ef[(e, f)] += c
                    count_f[f] += c

        #  M step: normalize
        for (e, f) in count_ef.keys():
            t[(e, f)] = count_ef[(e, f)] / count_f[f]

    return t



t = model1_train_e2f(bitext, opts) if opts.trainDirection != 'f2e' else model1_train_f2e(bitext, opts)

if opts.trainDirection != 'f2e':
    t = model1_train_e2f(bitext, opts)

    # Get the alignments
    for ff, ee in bitext:
            for (i, f_i) in enumerate(ff):
                max_p = float(0)
                best_j = 0
                for (j, e_j) in enumerate(ee):
                    if t[(f_i, e_j)] > max_p:
                        max_p = t[(f_i, e_j)]
                        best_j = j
                sys.stdout.write("%i-%i " % (i, best_j))
            sys.stdout.write("\n")

else:
    t = model1_train_f2e(bitext, opts)
    # Get the alignments
    for ff, ee in bitext:
        for (j, e_j) in enumerate(ee):
            max_p = float(0)
            best_i = 0
            for (i, f_i) in enumerate(ff):
                if t[(e_j, f_i)] > max_p:
                    max_p = t[(e_j, f_i)]
                    best_i = i
            sys.stdout.write("%i-%i " % (best_i, j))
        sys.stdout.write("\n")

end = time.time()


sys.stderr.write("\n...Done. Time elapsed: %.2f" % (end - start) + "s\n")
