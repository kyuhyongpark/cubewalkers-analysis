# Get Hamming distance in time in both updates starting from perturbation of distance 1
# for all models in the Cell Collective

import cupy as cp
import cubewalkers as cw
from cubewalkers.conversions import *
from cana.datasets.bio import load_all_cell_collective_models

WALKERS = 1000000
N_threshold = 100
T_synch = 3
FILE_NAME = "hamming_distance_cell_coll.txt"

nets = load_all_cell_collective_models()
fp = open(FILE_NAME, "w")

for net in nets:
    N = net.Nnodes
    # skip models with more than N_threshold
    if N > N_threshold:
        continue

    print(net.name)
    fp.write(net.name + '\n')
    print('N = ', N)
    fp.write('N = '+str(N)+'\n')

    rules = network_rules_from_cana(net)
    mymodel = cw.Model(rules=rules)

    # Get Hamming distances in synchronous update
    print('synchronous')
    fp.write('synchronous\n')
    mymodel.n_time_steps = T_synch
    mymodel.n_walkers = WALKERS // N
    derrida_synch = cp.zeros((mymodel.n_time_steps+1))
    for node in mymodel.vardict:
        di = mymodel.dynamical_impact(source_var=node,maskfunction=cw.update_schemes.synchronous,threads_per_block=(16,16))
        di = cp.sum(di, axis=1)
        derrida_synch += di
    derrida_synch /= N
    for derr in derrida_synch:
        print(derr)
        fp.write(str(derr) + '\n')

    # Get Hamming distances in asynchronous update
    print('asynchronous')
    fp.write('asynchronous\n')
    mymodel.n_time_steps = T_synch * N
    mymodel.n_walkers = WALKERS // N
    derrida_asynch = cp.zeros((mymodel.n_time_steps+1))
    for node in mymodel.vardict:
        di = mymodel.dynamical_impact(source_var=node,maskfunction=cw.update_schemes.asynchronous,threads_per_block=(16,16))
        di = cp.sum(di, axis=1)
        derrida_asynch += di
    derrida_asynch /= N
    for derr in derrida_asynch:
        print(derr)
        fp.write(str(derr) + '\n')
    print('- - - - - -')
    fp.write('\n')
