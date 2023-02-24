from io import StringIO
import cubewalkers as cw
import cupy as cp
from copy import deepcopy

DEBUG_USING_SHORT_TIME = False # IF TRUE, OUTPUTS WILL NOT HAVE TIME TO CONVERGE; DO NOT MODIFY
GLOBAL_WALKER_COUNT = 2500
GLOBAL_TPB = (16,16)
COMBINATIONS_TO_SIMULATE = { # WARNING: EACH COMBINATION TAKES A LONG TIME
    ('fuzzy','sourceless'),
    ('nonfuzzy','sourceless'),
    ('fuzzy','withsource'),
    ('nonfuzzy','withsource'),
}

def sourceless_quasicoherence(model, T_sample, maskfunction, fuzzy):
    c = 0
    n_core = 0
    n_source = 0
    n_all = 0
    for rule in StringIO(model.rules):
        name, func = map(lambda x: x.strip(), rule.split(','))
        if name == func:
            n_source += 1
            continue
        n_core += 1
        c += model.source_quasicoherence(name,
                         T_sample = T_sample,
                         fuzzy_coherence = fuzzy,
                         maskfunction = maskfunction,
                         threads_per_block = GLOBAL_TPB)
    
    n_all = n_source + n_core
    
    c = c/n_core # avg agreement over all perturbed source nodes
    
    if fuzzy: # then we need to renormalize the agreement fraction
        disagreement = 1-c # averaged over ALL nodes
        disagreement *= n_all/n_core # now averaged only over CORE nodes (source nodes always agree here)
        c = 1 - disagreement # now back to agreement, properly normalized
    
    return c

def simulate_model_coherence(models, sync, fuzzy, sourceless):
    coherences = {}
    W = GLOBAL_WALKER_COUNT
    for model_idx, (model_name, model) in enumerate(models.items()):
        model.n_walkers = W
        N = model.n_variables
        
        if not DEBUG_USING_SHORT_TIME:
            timescale = 2*N
            T = N**2 + 5 * timescale
            T_window = 5 * timescale
        else:
            T = 10
            T_window = 5
        
        model.n_time_steps = T

        print(f"Simulating Model {model_name} ({W=},{T=},{N=}). . .")
        if sync:
            maskfunction = cw.update_schemes.synchronous
        else:
            maskfunction = cw.update_schemes.asynchronous
        
        if sourceless:
            coherences[model_name] = sourceless_quasicoherence(model, T_window, maskfunction, fuzzy)
        else:
            coherences[model_name] = model.quasicoherence(T_sample=T_window,
                                                      fuzzy_coherence=fuzzy,
                                                      maskfunction=maskfunction,
                                                      threads_per_block=GLOBAL_TPB)

        print(
            f"Progress: {(model_idx+1)}/{total_models},\t coherence: {coherences[model_name]}")
    return coherences

IMPORT_RULES_FROM_FILES = True
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

for fuzzy_str, source_str in COMBINATIONS_TO_SIMULATE:
    fuzzy = (fuzzy_str == 'fuzzy')
    sourceless = (source_str == 'sourceless')
    
    filestring = f'./data/quasicoherence_{fuzzy_str}_{source_str}.csv'
    
    if fuzzy and sourceless:
        header = 'model name, SFQCNS, AFQCNS\n'
    if not fuzzy and sourceless:
        header = 'model name, SQCNS, AQCNS\n'
    if fuzzy and not sourceless:
        header = 'model name, SFQC, AFQC\n'
    if not fuzzy and  not sourceless:
        header = 'model name, SQC, AQC\n'
        
    sync_coherences=simulate_model_coherence(sync_models,True,fuzzy,sourceless)
    async_coherences=simulate_model_coherence(async_models,False,fuzzy,sourceless)
    
    with open(filestring,'w') as f:
        f.write(header)
        for model_name, smodel, amodel in [(k,sync_models[k],async_models[k]) for k in sorted(async_models)]:
            f.write((
                f'{model_name},'
                f'{cp.round(sync_coherences[model_name],6)},'
                f'{cp.round(async_coherences[model_name],6)}\n'
                ))