from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import AnnealSolution

data = np.loadtxt('reduce.out')

d = []
for i in data:
    s = AnnealSolution.AnnealSolution(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
    d.append(s)

minv = min([s.v for s in d])
minmatch = min([s.n_entry[6] for s in d])

sols = []
for s in d:
    if s.v < 2 * minv and s.n_entry[6] <= minmatch + 5:
        sols.append(s)
print(len(sols))

import time

ts1 = time.time()

solred = set()
for i in sols:
    solred.add(i)
for i in solred:
    print(i.n_entry)

ts2 = time.time()
print('time1core', ts2 - ts1)


