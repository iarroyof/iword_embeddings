import indexing
from joblib import Parallel, delayed
import logging
from time import gmtime, strftime
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import time
pyVersion = sys.version.split()[0].split(".")[0]
if pyVersion == '2':
    import cPickle as pickle
else:
    import _pickle as pickle

from scipy.sparse import coo_matrix, hstack, vstack


from pdb import set_trace as st

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def word_sparse_centroid(index_db, idf_model, word, vsize):

    try:
        for n, window in enumerate(index_db.windows(word)):
            sparse_embedding = idf_model.transform([" ".join(window)]) # .toarray()
            if n != 0:
                sparse_centroid = sparse_centroid + (sparse_centroid - sparse_embedding) / (n + 1)
            else:
                sparse_centroid = sparse_embedding

    except:
        return -1

    return coo_matrix(sparse_centroid, shape=(1, vsize))

class streamer(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        for s in open(self.file_name):
            yield s.strip()


def batches(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        chunk = chain([first], islice(iterator, size - 1))
        yield vstack([c for c in chunk])


parser = argparse.ArgumentParser(description='Build word datatabase and indexing sliding windows from plain text input file.')
parser.add_argument('--input', required=True, help='An input plain text file name.')
parser.add_argument('--output', default=None, help='An ouput database file name.')
parser.add_argument('--samples', default=50, type=int, help="The number of samples per word (independent of the index created, which does not need to be computed again for different sample sizes).")
parser.add_argument('--wsize', default=10, type=int, help="The number of words per sliding window (independent of the index created, which does not need to be computed again for different sample sizes).")
parser.add_argument('--chunk', default=100000, type=int, help="The number of word coodinates to be loaded into memory before writing them to the database.")
parser.add_argument('--test_words', default=None, help="Get test windows for these words.")
parser.add_argument('--idf', help="The IDF model to transform sliding windows into sparse embeddings.", default=None)
parser.add_argument('--dim', default=100, type=int, help="The dimensionality of"
                    " the output word embeddings.")

args = parser.parse_args()
if args.output is None:
    outputf = args.input + ".db"
else:
    outputf = args.output

logging.info("Starting to build index")
i = indexing.file_index(input_file = args.input, #'/almac/ignacio/data/INEXQA2012corpus/wikiEn_sts_clean_ph2.txt',
                        index_file = outputf,#'/almac/ignacio/data/INEXQA2012corpus/wikiEn_sts_clean_ph2.db',
                        mmap=True, wsize=args.wsize, sampsize=args.samples, n_jobs=1,
                        chunk_size=args.chunk)

i.fit()
logging.info("Index fitted!!")
logging.info("Output database: {}".format(outputf))

#if not args.test_words is None:
#    logging.info("Building example...")
#    for w in args.test_words.split(','):
#        print(i.windows(w))

try:
    if not args.idf is None:
        logging.info("Loading global TFIDF weights from: %s ..." % args.idf)
        with open(args.idf, 'rb') as f:
            if pyVersion == '2':
                vectorizer = pickle.load(f)
            else:
                vectorizer = pickle.load(f, encoding = 'latin-1')
    else:
        vectorizer = TfidfVectorizer(min_df = 1,
                #ngram_range=(1, args.ngrams),
                #encoding = "latin-1",
                decode_error = "replace",
                lowercase = True,
                #binary = True,# if args.localw.startswith("bin") else False,
                sublinear_tf = True,# if args.localw.startswith("subl") else False,
                #stop_words = "english" if args.stop == 'ost' else None
                )
        logging.info("Fitting local TFIDF weights from: %s ..." % args.input)
        lines = streamer(args.input)
        vectorizer.fit(lines)
except:
    logging.info("IDF model file does not exist in: %s ..." % args.idf)
    exit()

sparse_centroids = (word_sparse_centroid(i, vectorizer, word,
                            len(vectorizer.vocabulary_)) for word in i.vocab)

from itertools import chain
from itertools import islice
from sklearn.decomposition import MiniBatchDictionaryLearning as NMF

nmf = NMF(n_components=args.dim, random_state=1, alpha=.1, n_jobs=4)
batch_size = 10

for batch in batches(sparse_centroids, batch_size):
    dictionary = nmf.partial_fit(batch.todense())

st()