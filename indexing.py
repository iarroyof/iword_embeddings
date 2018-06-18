from pdb import set_trace as st

import threading
import numpy as np
import logging
import os
from functools import partial
from joblib import Parallel, delayed
from random import shuffle
import sqlite3
import os

class file_index(object):
    "Use n_jobs = 1 for now."
    def __init__(self, input_file, index_file=None, mmap=True, wsize=10,
                        encoding='latin1', sampsize=50, n_jobs=1, chunk_size=1000, verbose=True):

        self.mmap = mmap
        self.memory = ":memory:"
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
        logging.info("Got coordinates for '%s'\n" % word)

        windows = []
        for r, w in coordinates:
            try:
                ln = self.index_lines[r].split()
            except UnicodeDecodeError:
                continue
            except AttributeError:
                print("\nCall 'load_input()' method before querying windows.\n")
                raise


            start = min(len(ln[0:w]), self.wsize)
            windows.append(ln[w - start:w] + ln[w + 1:w + (self.wsize + 1)])

        logging.info("Got windows for '%s'\n" % word)
        return windows

    def fit(self):
        f = open(self.input_file, encoding='latin-1')
        if self.n_jobs > 1 or self.n_jobs == -1:
            #Parallel(n_jobs=self.n_jobs)(delayed(self.index_row)(n, row, self.conn)
            #                                    for n, row in enumerate(f))
            assert self.index_file  # Index file must be specified in multithreading mode
            for n, row in enumerate(f):
                t = InsertionThread(n, row, self.index_file)
                t.start()
        else:
            if self.index_file != self.memory and self.chunk_size > 0:
                c = 0
                ck = 0

                for n, row in enumerate(f):
                    self.index_row(n, row)
                    if c == self.chunk_size:
                        c = 0
                        self.conn.commit()
                        logging.info("Saved index chunk %d into index file %s \n" % (ck, self.index_file))
                        ck += 1
                    c += 1
                
            else:
                logging.info("Creating index in-memory database... \n")
                for n, row in enumerate(f):
                    self.index_row(n, row)

        try:
            self.cursor.execute("create index idxword on words(word)")
            self.conn.commit()
            logging.info("Saved index into index file datbase %s\n" % self.index_file)
            return self
        except:
            print("Database couldn't be created... EXIT error.")
            raise


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

        for of, word in enumerate(row.split()):
            t = (word, self.tup2str((line_id, of)) )
            insert = "INSERT INTO words VALUES (?, ?)"
            try:
                cursor.execute(insert, t)
            except sqlite3.OperationalError:
                print("Problems to create word table '%s'.\n" % word)
                self.disconnect()
                raise

        if self.n_jobs != 1 and self.n_jobs != 0:
            self.conn.commit()


    def load_input(self):
        """ Call this method when a prefitted index db file already exists"""
        with open(self.input_file, encoding=self.encoder) as fc:
            self.index_lines = fc.readlines()
        # Return pointer to the index
        return self


class InsertionThread(threading.Thread):

    def __init__(self, line_id, row, filename):
        super(InsertionThread, self).__init__()
        self.row = row
        self.line_id = line_id
        self.filename = filename

    def run(self):
        conn = sqlite3.connect(self.filename, timeout=10, check_same_thread=False)
        #conn.execute('CREATE TABLE IF NOT EXISTS threadcount (threadnum, count);')
        for of, word in enumerate(self.row.split()):
            t = (self.line_id, of)
            create = """CREATE TABLE IF NOT EXISTS "{}" (row int, pos int)""".format(word) 
            insert = """INSERT INTO "{}" VALUES (?,?)""".format(word)
            try:
                conn.execute(create)
                conn.execute(insert, t)
            except sqlite3.OperationalError:
                print("Problems to create word table '%s'.\n" % word)
                conn.commit()
                conn.close()
                raise
            
        conn.commit()
