# Get Hamming distance in time in both updates starting from perturbation of distance 1

import cupy as cp
import cubewalkers as cw
import pystablemotifs as sm

MODEL = '../models/Reduced_tumour_model_no_sink_E1D1.txt'
WALKERS = 1000000
T_synch = 10
FILE_NAME = "hamming_distance.txt"

fp = open(FILE_NAME, "w")

rules = sm.format.remove_comment_lines(open(MODEL))
mymodel = cw.Model(rules=rules)

print(MODEL)
fp.write(MODEL + '\n')
N = mymodel.n_variables
print('N = ', N)
fp.write('N = '+str(N)+'\n')

# Get Hamming distances in synchronous update
print('synchronous')
fp.write('synchronous\n')
mymodel.n_time_steps = T_synch
mymodel.n_walkers = WALKERS // N
derrida_synch = cp.zeros((mymodel.n_time_steps+1))
for node in mymodel.vardict:
    di = mymodel.dynamical_impact(source_var=node,maskfunction=cw.update_schemes.synchronous)
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
    di = mymodel.dynamical_impact(source_var=node,maskfunction=cw.update_schemes.asynchronous)
    di = cp.sum(di, axis=1)
    derrida_asynch += di
derrida_asynch /= N
for derr in derrida_asynch:
    print(derr)
    fp.write(str(derr) + '\n')
print('- - - - - -')
fp.write('\n')
