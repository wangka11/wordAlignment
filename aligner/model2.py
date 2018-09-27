#!/usr/bin/env python
import optparse
import sys
import time
from nltk.stem import SnowballStemmer
from model1 import *

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-k", "--iteration", dest="iterations", default=5, help="Number of iterations (default=10)")
optparser.add_option("-s", "--stemming", dest="stemming", default=True, help="Word stemming (default=true)")
# optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with IBM2 and EM algorithm...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

if opts.stemming:
    stemmer_fr = SnowballStemmer("french")
    stemmer_en = SnowballStemmer("english")

    stemmed_text = []
    for ff, ee in bitext:

        stemmed_text.append([[stemmer_fr.stem(item.decode("utf-8")) for item in ff],
                                [stemmer_en.stem(item) for item in ee]])
    bitext = stemmed_text

start = time.time()


q = defaultdict(float)
t = model1_train_f2e(bitext, opts)
for ff, ee in bitext:
    l, m = len(ee), len(ff)
    for j, e in enumerate(ee):
        for i, f in enumerate(ff):
            q[(j, i, l, m)] = float(1.0) / (m + 1)

k = 0
while k < opts.iterations:
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
            cur = t[(e, f)] * q[(j, i, l, m)]
            if cur > max_p:
                max_p = cur
                best_i = i
        sys.stdout.write("%i-%i " % (best_i, j))
    sys.stdout.write("\n")


end = time.time()
sys.stderr.write("\n...Done. Time elapsed: %.2f" % (end - start) + "s\n")