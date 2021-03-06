import indexing
import logging
from time import gmtime, strftime
import argparse


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='Build word datatabase and indexing sliding windows from plain text input file.')
parser.add_argument('--input', required=True, help='An input plain text file name.')
parser.add_argument('--output', default=None, help='An ouput database file name.')
parser.add_argument('--samples', default=50, type=int, help="The number of samples per word (independent of the index created, which does not need to be computed again for different sample sizes).")
parser.add_argument('--wsize', default=10, type=int, help="The number of words per sliding window (independent of the index created, which does not need to be computed again for different sample sizes).")
parser.add_argument('--chunk', default=100000, type=int, help="The number of word coodinates to be loaded into memory before writing them to the database.")
parser.add_argument('--test_words', default=None, help="Get test windows for these words.")
                                                            
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

if not args.test_words is None:
    logging.info("Building example...")
    for w in args.test_words.split(','):
        print(i.windows(w))

