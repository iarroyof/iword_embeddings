import indexing
import numpy as np
from joblib import Parallel, delayed
import logging, os
from time import gmtime, strftime
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import Birch
from itertools import chain
from itertools import islice
from functools import partial
import sys
import time
pyVersion = sys.version.split()[0].split(".")[0]
if pyVersion == '2':
    import cPickle as pickle
else:
    import _pickle as pickle

from scipy.sparse import coo_matrix, csr_matrix, vstack
from pdb import set_trace as st

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class wordCentroids(object):
    def __init__(self, db, vect):
        self.db = db
        self.vect = vect
        self.word = None

    def __iter__(self):
        vocab_size = len(self.vect.vocabulary_)
        for word in self.vect.vocabulary_:
            
            yield word, word_sparse_centroid(self.db, self.vect, word, vocab_size)
            

def word_sparse_centroid(index_db, idf_model, word, vsize):
    try:
        windows = index_db.windows(word.encode())
        if len(windows) == 1:
            return coo_matrix(idf_model.transform([b' '.join(windows[0])]), shape=(1, vsize))
            
        for n, window in enumerate(windows):
            sparse_embedding = idf_model.transform([b' '.join(window)])
            if n != 0:
                sparse_centroid = sparse_centroid + (sparse_embedding - sparse_centroid) / (n + 1)
            else:
                sparse_centroid = sparse_embedding

        return coo_matrix(sparse_centroid, shape=(1, vsize))

    except:
        logging.info("Word not found in index DB: %s ...\nType q + Enter and fix the issue..." % word.encode())
        return None


class streamer(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        for s in open(self.file_name, 'rb'):
            yield s.strip()


def batches(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        chunk = chain([first], islice(iterator, size - 1))
        chuchu = [c for c in chunk]
        yield [w for w, c in chuchu], vstack([c for w, c in chuchu])


parser = argparse.ArgumentParser(description='Build word embeddings by using clustering based sparse coding.')
parser.add_argument('--input', required=True, help='An input plain text file name.')
parser.add_argument('--output', default="dictionary_learned.vec", help='The output text file name containing word vectors in word2vec format.')
parser.add_argument('--db', default=None, help='An ouput database file name.')
parser.add_argument('--samples', default=50, type=int, help="The number of samples per word (independent of the index" 
                                            " created, which does not need to be computed again for different sample sizes).")
parser.add_argument('--wsize', default=10, type=int, help="The number of words per sliding window (independent of the index" 
                                            " created, which does not need to be computed again for different sample sizes).")
parser.add_argument('--chunk', default=100000, type=int, help="The number of word coodinates to be loaded into memory before" 
                                                                                            " writing them to the database.")
parser.add_argument('--idf', help="The IDF model to transform sliding windows into sparse embeddings.", default=None)
parser.add_argument('--dim', default=100, type=int, help="The dimensionality of the output word embeddings.")
parser.add_argument('--bsize', default=100, type=int, help="The number of samples per batch for minibatch clustering.")
parser.add_argument('--verbo', default=False, type=int, help="Verbose: setting it greater than 10 prints detailed reports.")
args = parser.parse_args()

if args.dim > args.bsize:
    logging.info("The batch size must be greater or equal than dimensionality (number of clustering centroids)")
    exit()

batch_size = args.bsize

if args.db is None:
    outputf = args.input + ".db"
else:
    outputf = args.db

try:
    if not args.idf is None:
        logging.info("Loading global TFIDF weights from: %s ..." % args.idf)
        with open(args.idf, 'rb') as f:
            if pyVersion == '2':
                vectorizer = pickle.load(f)
            else:
                vectorizer = pickle.load(f, encoding = 'latin-1')
    else:
        vectorizer = TfidfVectorizer(
                #ngram_range=(1, args.ngrams),
                #encoding = "latin-1",
                decode_error = "replace",
                lowercase = True,
                #binary = True,# if args.localw.startswith("bin") else False,
                sublinear_tf = True,# if args.localw.startswith("subl") else False,
                stop_words = "english" #if args.stop == 'ost' else None
                )
        logging.info("Fitting local TFIDF weights from: %s ..." % args.input)
        lines = streamer(args.input)
        vectorizer.fit(lines)
        
except:
    logging.info("IDF model file does not exist in: %s ..." % args.idf)
    exit()

DBexists = os.path.exists(outputf)
logging.info("Instantiating index object...")
index = indexing.file_index(input_file = args.input, #'/almac/ignacio/data/INEXQA2012corpus/wikiEn_sts_clean_ph2.txt',
                        index_file = outputf, vectorizer=vectorizer,
                        mmap=True, wsize=args.wsize, sampsize=args.samples, n_jobs=1,
                        chunk_size=args.chunk,
                        verbose=args.verbo)
if not DBexists:
    logging.info("Starting to build index into DB file %s" % outputf)
    index.fit()
    logging.info("Index fitted!!")
    logging.info("Output database: {}".format(outputf))
else:
    logging.info("Index loaded from DB file %s" % outputf)
    
if index.vocab_size < args.bsize:
    logging.info("ERROR: Batch size [{}] must be greater than vocabulary [{}]".format(args.bsize, index.vocab_size))
    exit()

sparse_word_centroids = wordCentroids(db=index, vect=vectorizer)
# Tal vez pueda cargar la matrix dipersa de word_centroids en ram y hacer NMF.
# 
logging.info("Fitting Birch clustering for sparse coding ...")
birch = Birch(threshold=0.5, branching_factor=50, n_clusters=args.dim) #MiniBatchKMeans(n_clusters=args.dim, init='k-means++', max_iter=4, batch_size=batch_size)
words = []
for i, batch in enumerate(batches(sparse_word_centroids, batch_size)):
    #buffer.append(vstack(batch))
    logging.info("Fitted the %d th batch..." % i)
    words.append(batch[0])
    birch.partial_fit(batch[1])

words = list(chain(*words))

for i, batch in enumerate(batches(sparse_word_centroids, batch_size)):
    if i == 0:
        #word_embeddings = batch[1].dot(csr_matrix(birch.subcluster_centers_).T)
        word_embeddings = birch.transform(batch[1])
    else:
        #word_embeddings = vstack([word_embeddings, batch[1].dot(csr_matrix(birch.subcluster_centers_).T)])
        
        word_embeddings = np.vstack([word_embeddings, birch.transform(batch[1])])

# word_embeddings.shape = (vocab_size, args.dim)
logging.info("DB Vocabulary size %d ..." % index.vocab_size)
logging.info("Vectorizer vocabulary size %d ..." % len(vectorizer.vocabulary_.keys()))
logging.info("Shape of resulting embedding matrix: ({}, {})".format(birch.subcluster_centers_.shape[0],
                                                                    birch.subcluster_centers_.shape[1]))
logging.info("Writing word vectors into file %s ..." % args.output)
write = partial(indexing.write_given_embedding, fname=args.output)
                                                
with open(args.output, "w") as f:
    f.write("{} {}\n".format(word_embeddings.shape[0], word_embeddings.shape[1]) )

    Parallel(#backend='threading',
        n_jobs=20
            )(delayed(write)(w, e)
            for w, e in zip(words, word_embeddings.toarray()
                if isinstance(word_embeddings, csr_matrix)
                else word_embeddings))

logging.info("Embedding process has FINISHED!!")

