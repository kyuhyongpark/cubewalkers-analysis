# Get Derrida coefficient in sychronous and asynchronous updates

import cupy as cp
import cubewalkers as cw
from cubewalkers.conversions import *
from cana.datasets.bio import load_all_cell_collective_models

WALKERS = 1000000
N_threshold = 10
FILE_NAME = "derrida_cell_coll.txt"

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

    # Get Derrida coefficient in synchronous update
    print('synchronous')
    fp.write('synchronous\n')
    mymodel.n_walkers = WALKERS
    derrida_synch = mymodel.derrida_coefficient(threads_per_block=(16,16))
    print(derrida_synch)
    fp.write(str(derrida_synch) +'\n')

    # Get Derrida coefficient in asynchronous update
    print('asynchronous')
    fp.write('asynchronous\n')
    mymodel.n_time_steps = N
    mymodel.n_walkers = WALKERS // N
    derrida_asynch = cp.zeros((N+1))
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
