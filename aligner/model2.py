#!/usr/bin/env python
from model1 import *


def model2_train_f2e(bitext, opts):

    q = defaultdict(float)
    t = model1_train_f2e(bitext, opts)
    for ff, ee in bitext:
        l, m = len(ee), len(ff)
        for j, e in enumerate(ee):
            for i, f in enumerate(ff):
                #q[(j, i, l, m)] = random.uniform(0, 1)

                q[(j, i, l, m)] = float(1.0) / (m + 1)

    k = 0
    while k < int(opts.iterations):
        k += 1
        # Initialize all the counts
        count_ef, count_f, count_ji, count_j = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        denominator = defaultdict(float)
        for ff, ee in bitext:
            l, m = len(ee), len(ff)

            # Get the denominator of the theta
            for j, e in enumerate(ee):
                denominator[e] = 0.0
                for i, f in enumerate(ff):
                    denominator[e] += t[(e, f)] * q[(j, i, l, m)]
            for j, e in enumerate(ee):
                for i, f in enumerate(ff):
                    theta = (t[(e, f)] * q[(j, i, l, m)]) / denominator[e]
                    count_ef[(e, f)] += theta
                    count_f[f] += theta
                    count_ji[(j, i, l, m)] += theta
                    count_j[(j, l, m)] += theta

        for e, f in t.keys():
            t[(e, f)] = float(count_ef[(e, f)]) / count_f[f]

        for ff, ee in bitext:
            l, m = len(ee), len(ff)
            for j, e in enumerate(ee):
                for i, f in enumerate(ff):
                    q[(j, i, l, m)] = count_ji[(j, i, l, m)] / count_j[(j, l, m)]


    for ff, ee in bitext:
        l, m = len(ee), len(ff)
        for j, e in enumerate(ee):
            max_p = 0
            best_i = 0
            for i, f in enumerate(ff):
                p = t[(e, f)] * q[(j, i, l, m)]
                if p > max_p:
                    max_p = p
                    best_i = i
            sys.stdout.write("%i-%i " % (best_i, j))
        sys.stdout.write("\n")


def model2_train_e2f(bitext, opts):
    q = defaultdict(float)
    t = model1_train_e2f(bitext, opts)
    for ff, ee in bitext:
        l, m = len(ee), len(ff)
        for i, f in enumerate(ff):
            for j, e in enumerate(ee):

                q[(i, j, l, m)] = float(1.0) / (l + 1)

    k = 0
    while k < int(opts.iterations):
        k += 1
        # Initialize all the counts
        count_fe, count_e, count_ij, count_i = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(
            float)
        denominator = defaultdict(float)
        for ff, ee in bitext:
            l, m = len(ee), len(ff)

            # Get the denominator of the theta
            for i, f in enumerate(ff):
                denominator[f] = 0.0
                for j, e in enumerate(ee):
                    denominator[f] += t[(f, e)] * q[(i, j, l, m)]
            for i, f in enumerate(ff):
                for j, e in enumerate(ee):
                    theta = (t[(f, e)] * q[(i, j, l, m)]) / denominator[f]
                    count_fe[(f, e)] += theta
                    count_e[e] += theta
                    count_ij[(i, j, l, m)] += theta
                    count_i[(i, l, m)] += theta

        for f, e in t.keys():
            t[(f, e)] = float(count_fe[(f, e)]) / count_e[e]

        for ff, ee in bitext:
            l, m = len(ee), len(ff)
            for i, f in enumerate(ff):
                for j, e in enumerate(ee):
                    q[(i, j, l, m)] = count_ij[(i, j, l, m)] / count_i[(i, l, m)]

    for ff, ee in bitext:
        l, m = len(ee), len(ff)
        for i, f in enumerate(ff):
            max_p = 0
            best_j = 0
            for j, e in enumerate(ee):
                p = t[(f, e)] * q[(i, j, l, m)]
                if p > max_p:
                    max_p = p
                    best_j = j
            sys.stdout.write("%i-%i " % (i, best_j))
        sys.stdout.write("\n")
