import time
import numpy as np
import csv
from pdb import set_trace as st


in_file = "sample.txt" # "/almac/ignacio/data/mexicoSp/cscm_Text_utf8_min.txt"
start = time.time()
def elapsed():
    return time.time() - start

# count data rows, to preallocate array
f = open(in_file, 'rb')
def count(f):
    while 1:
        block = f.read(100)  # 65536)
        if not block:
             break
        yield block.count(str.encode('\n'))

linecount = sum(count(f))
print ('\n%.3fs: file has %s rows' % (elapsed(), linecount))

# pre-allocate array and load data into array
m = np.zeros(linecount, dtype=[('a', np.uint32), ('b', np.uint32)])
f.seek(0)

with open(in_file, 'rb') as f:
    file_byte = []
    bytes = []
    while True:
    #block = f.read(100)  # 65536)
        byte = f.read(1)
    #st()
        if not byte:
            break
        elif byte != b'\n': # ord('\n'):
        #byte = f.read(1)
            bytes.append(byte)
        else:
            file_byte.append(b"".join(bytes))
            bytes = []

file_list = [b.decode("utf-8") for b in file_byte]
st()
f = open(in_file, 'rb')
for i, row in enumerate(f):
    m[i] = int(row[0]), int(row[1])

print ('%.3fs: loaded' % elapsed())
# sort in-place
m.sort(order='b')

print ('%.3fs: sorted' % elapsed())
