#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
from model2 import *
from nltk.stem import SnowballStemmer
import time

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-k", "--iteration", dest="iterations", default=10, help="Number of iterations (default=10)")
optparser.add_option("-s", "--stemming", dest="stemming", default=True, help="Word stemming (default=true)")
optparser.add_option("-t", "--trainDirection", dest="trainDirection", default="f2e", help="Translation direction (default=f2e): e2f or f2e")
# optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)


bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

stemming = False if str(opts.stemming).lower() == "false" else True
if stemming:
    stemmer_fr = SnowballStemmer("french")
    stemmer_en = SnowballStemmer("english")

    stemmed_text = []
    for ff, ee in bitext:

        stemmed_text.append([[stemmer_fr.stem(item.decode("utf-8")) for item in ff],
                                [stemmer_en.stem(item.decode("utf-8")) for item in ee]])
    bitext = stemmed_text

start = time.time()


if __name__ == "__main__":
    sys.stderr.write("Training with IBM Model2 and EM algorithm...")
    start = time.time()
    if (opts.trainDirection == "f2e"):
        model2_train_f2e(bitext, opts)
    else:
        model2_train_e2f(bitext, opts)
    end = time.time()
    sys.stderr.write("\n...Done. Time elapsed: %.2f" % (end - start) + "s\n")