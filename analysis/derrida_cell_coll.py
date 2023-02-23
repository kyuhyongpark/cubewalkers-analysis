# Get Derrida coefficient in sychronous and asynchronous updates

import cubewalkers as cw
import cupy as cp
from copy import deepcopy
from io import StringIO

NUMBER_OF_WALKERS = 100000
IMPORT_RULES_FROM_FILES = True

def derrida_cell_coll(models, sync=True ,W=NUMBER_OF_WALKERS):
    derrida_coefficients = {}
    for model_idx, (model_name, model) in enumerate(models.items()):
        N = model.n_variables
        
        if sync:
            print(f"Calculating Derrida coefficient in synchronous for {model_name} ({N=},{W=}). . .")
            model.n_walkers = W
            derrida_coefficients[model_name] = float(model.derrida_coefficient(threads_per_block=(16,16)))
        else:
            print(f"Calculating Derrida coefficient in asynchronous for {model_name} ({N=},{W=}). . .")
            model.n_time_steps = N
            model.n_walkers = W // N
            derrida_asynch = cp.zeros((N+1))
            for node in model.vardict:
                di = model.dynamical_impact(source_var=node,maskfunction=cw.update_schemes.asynchronous,threads_per_block=(16,16))
                di = cp.sum(di, axis=1)
                derrida_asynch += di
            derrida_asynch /= N
            derrida_coefficients[model_name] = float(derrida_asynch[-1])
            
        print(f"Progress: {(model_idx+1)}/{total_models}")
    return derrida_coefficients

def sourceless_derrida(model,sync=True,W=NUMBER_OF_WALKERS):
    if sync:
        maskfunction = cw.update_schemes.synchronous
        T = 1
        model.n_walkers = W
    else:
        maskfunction = cw.update_schemes.asynchronous
        T = model.n_variables
        model.n_walkers = W // T
    n_ns = 0
    di = 0
    for rule in StringIO(model.rules):
        varname, func = rule.split(',')
        func = func.strip()
        if varname == func:
            continue
        n_ns += 1
        di += cp.sum(
            model.dynamical_impact(
                varname, 
                n_time_steps = T, 
                maskfunction=maskfunction,
                threads_per_block=(16,16)),
            axis=1)
    
    return di[-1]/n_ns

def derrida_cell_coll_sourceless(models, sync=True ,W=NUMBER_OF_WALKERS):
    derrida_coefficients_sourceless = {}
    for model_idx, (model_name, model) in enumerate(models.items()):
        derrida_coefficients_sourceless[model_name] = sourceless_derrida(model,sync=sync,W=W)
        print(f"Progress: {(model_idx+1)}/{total_models}")
    return derrida_coefficients_sourceless

cc_models_dir = './models/cell_collective/'
if IMPORT_RULES_FROM_FILES:
    from os import listdir
    
    sync_models = {}
    for fname in listdir(cc_models_dir):
        with open(cc_models_dir+fname) as rulefile:
            name = fname.strip('.txt')
            rules = rulefile.read()
            sync_models[name]=cw.Model(rules)
else:
    from cana.datasets.bio import load_all_cell_collective_models
    def cell_collective_models():
        return {BN.name:cw.Model(cw.conversions.network_rules_from_cana(BN)) 
                for BN in load_all_cell_collective_models()}
    sync_models = cell_collective_models()
    for name,model in sync_models.items():
        with open(cc_models_dir+name+'.txt','w') as rulefile:
            rulefile.write(model.rules)

total_models = len(sync_models)
async_models = deepcopy(sync_models)

dc_sync = derrida_cell_coll(sync_models,sync=True)
dc_async = derrida_cell_coll(sync_models,sync=False)
dcns_sync = derrida_cell_coll_sourceless(sync_models,sync=True)
dcns_async = derrida_cell_coll_sourceless(async_models,sync=False)

with open('./data/derrida_coefficients.csv','w') as f:
    f.write(('model name, '
             'synchronous DC, ' 
             'asynchronous DC, '
             'synchronous DC (sourceless), '
             'asynchronous DC (sourceless)\n'))
    for model_name in sorted(sync_models):
        f.write((f'{model_name},'
                f'{dc_sync[model_name]},'
                f'{dc_async[model_name]},'
                f'{dcns_sync[model_name]},'
                f'{dcns_async[model_name]}\n'))