# Get the Hamming distance in time in both updates
# for all models in the Cell Collective.
# Starting from a perturbation of Hamming distance 1,
# the states are updated for 10 timesteps in synchronous,
# and 10 N timesteps in asynchronous.

from io import StringIO
import cupy as cp
import cubewalkers as cw

IMPORT_RULES_FROM_FILES = True
cc_models_dir = './models/cell_collective/'

DEBUG_USING_SHORT_TIME = True # IF TRUE, OUTPUTS WILL NOT HAVE TIME TO CONVERGE; DO NOT MODIFY
GLOBAL_WALKER_COUNT = 2500
GLOBAL_TPB = (16,16)
T_SYNC = 10
COMBINATIONS_TO_SIMULATE = { # WARNING: EACH COMBINATION TAKES A LONG TIME
    ('sourceless'),
    ('withsource'),
}

def import_models(cc_models_dir, IMPORT_RULES_FROM_FILES = True):
    if IMPORT_RULES_FROM_FILES:
        from os import listdir
        
        models = {}
        for fname in listdir(cc_models_dir):
            with open(cc_models_dir+fname) as rulefile:
                name = fname.split(".")[0]
                rules = rulefile.read()
                models[name]=cw.Model(rules)
    else:
        from cana.datasets.bio import load_all_cell_collective_models
        def cell_collective_models():
            return {BN.name:cw.Model(cw.conversions.network_rules_from_cana(BN)) 
                    for BN in load_all_cell_collective_models()}
        models = cell_collective_models()
        for name,model in models.items():
            with open(cc_models_dir+name+'.txt','w') as rulefile:
                rulefile.write(model.rules)
    return models

def hamming_distances_in_time(model, maskfunction, sourceless=True):
    hds = cp.zeros((model.n_time_steps+1))
    n_core = 0
    n_all = model.n_variables
    for rule in StringIO(model.rules):
        name, func = map(lambda x: x.strip(), rule.split(','))
        if name == func and sourceless:
            continue
        n_core += 1
        di = model.dynamical_impact(source_var=name,maskfunction=maskfunction,threads_per_block=GLOBAL_TPB)
        hds += cp.sum(di, axis=1)
        
    hds /= n_core # avg agreement over all perturbed nodes
    
    return hds

def simulate_hamming_distances_in_time(models, sync, T_sync, sourceless):
    results = {}
    W = GLOBAL_WALKER_COUNT
    for model_idx, (model_name, model) in enumerate(models.items()):
        model.n_walkers = W
        N = model.n_variables

        if not DEBUG_USING_SHORT_TIME:
            if sync:
                T = T_sync
            else:
                T = T_sync * N
        else:
            T = 2
        model.n_time_steps = T

        print(f"Simulating Model {model_name} ({W=},{T=},{N=}). . .")
        if sync:
            maskfunction = cw.update_schemes.synchronous
        else:
            maskfunction = cw.update_schemes.asynchronous

        results[model_name] = hamming_distances_in_time(model, maskfunction, sourceless=sourceless)

        print(f"Progress: {(model_idx+1)}/{total_models}")
    return results

models = import_models(cc_models_dir, IMPORT_RULES_FROM_FILES=IMPORT_RULES_FROM_FILES)
total_models = len(models)

for source_str in COMBINATIONS_TO_SIMULATE:
    sourceless = (source_str == 'sourceless')
    
    filestring = f'./data/hamming_distances_in_time_{source_str}.csv'

    sync_hamming_distances = simulate_hamming_distances_in_time(models,True,T_SYNC,sourceless)
    async_hamming_distances = simulate_hamming_distances_in_time(models,False,T_SYNC,sourceless)

    with open(filestring,'w') as f:
        for model_name in sorted(models):
            shd = ','.join(map(lambda x: str(cp.round(x,6)),sync_hamming_distances[model_name]))
            ahd = ','.join(map(lambda x: str(cp.round(x,6)),async_hamming_distances[model_name]))
            f.write(f'{model_name},synchronous,{shd}\n')
            f.write(f'{model_name},asynchronous,{ahd}\n')