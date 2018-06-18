import indexing
import logging
from time import gmtime, strftime


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

logging.info("Starting to build index")
i = indexing.file_index(input_file='/almac/ignacio/data/INEXQA2012corpus/wikiEn_sts_clean_ph2.txt', 
                        index_file='/almac/ignacio/data/INEXQA2012corpus/wikiEn_sts_clean_ph2.db', 
                        mmap=True, wsize=10, sampsize=50, n_jobs=1, 
                        chunk_size=100000)

i.fit()
logging.info("Index fitted!!")
logging.info("Building example...")
print(i.windows('plasma'))
logging.info("Index fitted!!")
