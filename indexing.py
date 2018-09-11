from pdb import set_trace as st
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
import numpy as np
from scipy.sparse import csr_matrix
import logging
import os
from functools import partial
from joblib import Parallel, delayed
from random import shuffle
import sqlite3
import os


class file_index(object):
    "Use n_jobs = 1 for now."
    def __init__(self, input_file, index_file=None, mmap=True, wsize=10, vectorizer=None,
                        encoding='latin1', sampsize=50, n_jobs=1, chunk_size=1000, verbose=True):

        self.mmap = mmap
        self.memory = ":memory:"
        if not (vectorizer is None):
            self.vectorizer = vectorizer
            self.tokenizer = vectorizer.build_tokenizer()
        self.encoder = encoding
        self.index_file = self.memory if (not index_file or index_file == ":memory:") else index_file
        self.chunk_size = chunk_size
        self.input_file = input_file
        self.wsize = wsize
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.sampsize = sampsize
        if not os.path.exists(self.index_file):
            self.connect()
            self.cursor.execute("create table words (word text, coo text)")
        else:
            self.connect()
            self.load_input()
                                                

    def __enter__(self):
        self.connect()
        return self


    def __exit__(self):
        self.disconnect()
        return self


    def windows(self, word):
        if self.n_jobs != 1:
            self.connect()

        try:
            self.index_lines
        except AttributeError:
            self.load_input()

        if self.sampsize > 0:
            query = "select * from words where word=? order by random() limit ?"
            t = (word, self.sampsize)
            self.cursor.execute(query, t)
        else:
            query = "select * from words where word=?"
            self.cursor.execute(query, (word, ))

        coordinates = self.str2tup([t for w, t in self.cursor.fetchall()])

        windows = []
        for r, w in coordinates:
            try:
                ln = self.index_lines[r].split() #decode("utf-8").split()
            except UnicodeDecodeError:
                continue
            except AttributeError:
                print("\nCall 'load_input()' method before querying windows.\n")
                raise


            start = min(len(ln[0:w]), self.wsize)
            windows.append(ln[w - start:w] + ln[w + 1:w + (self.wsize + 1)])

        if self.verbose > 10:
            logging.info("Got windows for '%s'\n" % word)
        return windows


    def fit(self):
        with open(self.input_file, mode='rb') as f: # encoding='latin-1', mode='rb') as f:
            if self.index_file != self.memory and self.chunk_size > 0:
                c = 0
                ck = 0
                for n, row in enumerate(enumerate(f)):
                    #st()
                    self.index_row(n, row[1])
                    if c == self.chunk_size:
                        c = 0
                        self.conn.commit()
                        if self.verbose > 5:
                            logging.info("Saved index chunk %d into index file %s \n" % (ck, self.index_file))
                        ck += 1
                    c += 1

            else:
                if self.verbose:
                    logging.info("Creating index in-memory database... \n")

                for n, row in enumerate(get_binary(self.input_file)):
                    self.index_row(n, row)

            try:
                self.cursor.execute("create index idxword on words(word)")
                self.conn.commit()
            # Getting properties
                self.cursor.execute("SELECT * FROM words")
                self.vocab = list(set([r[0] for r in self.cursor.fetchall()]))
                self.vocab_size = len(self.vocab)
            
                if self.verbose:
                    logging.info("Saved index into index file datbase %s\n" % self.index_file)
                return self
            except:
                print("Database couldn't be created... EXIT error.")
                raise
            

    def load_input(self):
        """ Call this method when a prefitted index db file already exists"""
        with open(self.input_file, mode='rb') as fc: # encoding=self.encoder, mode='rb') as fc:
            self.index_lines = fc.readlines()

        self.cursor.execute("SELECT * FROM words")
        self.vocab = list(set([r[0] for r in self.cursor.fetchall()]))
        self.vocab_size = len(self.vocab)
        logging.info("Loaded index database properties and connections..")
        # Return pointer to the index
        return self


    def connect(self):
        self.conn = sqlite3.connect(self.index_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        return self


    def disconnect(self):
        self.conn.commit()
        self.conn.close()
        return self


    def tup2str(self, t):
        if isinstance(t, list):
            return [str(a) + ',' + str(b) for a, b in t]
        else:
            return str(t[0]) + ',' + str(t[1])


    def str2tup(self, t):
        if isinstance(t, list):
            r = []
            for x in t:
                r.append(self.str2tup(x))
            return r
        else:
            a, b = t.split(',')
            return (int(a), int(b))


    def index_row(self, line_id, row, conn=None):
        if self.n_jobs != 1 and self.n_jobs != 0:
            cursor = conn.cursor()
        else:
            cursor = self.cursor

        for of, word in enumerate(self.tokenize(row)):
            if word is None: continue
            t = (word, self.tup2str((line_id, of)) )
            insert = "INSERT INTO words VALUES (?, ?)"
            try:
                cursor.execute(insert, t)
            except sqlite3.OperationalError:
                print("Problems to create word table '%s'.\n" % word)
                self.disconnect()
                raise


    def tokenize(self, string):
        if self.tokenizer:
            if self.vectorizer.lowercase:
                try:
                    string = string.decode(errors="replace").lower()
                except Exception as e:
                    logging.info("Problems occurred while indexing row: {}\nEXCEPTION: {}".format(row, e))
                    return None
            return [w.encode() for w in self.tokenizer(string)]
        else:
            self.vectorizer = TfidfVectorizer()
            self.tokenizer = self.vectorizer.build_tokenizer()
            return self.tokenize(string)


def write_given_embedding(word, arr, fname):
    if isinstance(arr, csr_matrix):
        arr = arr.toarray().reshape(1,-1)[0]
        
    row_we = word + " " + " ".join(["{0:.6f}".format(i) for i in arr]) + '\n'
    with open(fname, "a") as f:
        f.write(row_we)


def write_embedding(word, embedding_matrix, centroid, fname):
    word_embedding = embedding_matrix.dot(centroid.T)
    write_given_embedding(word, word_embedding, fname)

