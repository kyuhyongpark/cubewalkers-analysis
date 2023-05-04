import cubewalkers as cw
import cupy as cp
from copy import deepcopy

IMPORT_RULES_FROM_FILES = True
CORRECTED_MODELS = True

if CORRECTED_MODELS:
    models_dir = './models/corrected_models/'
    IMPORT_RULES_FROM_FILES = True # The corrections are only available in files
    OutFileName = './data/corrected_models/converged_average_node_values.csv'
else:
    models_dir = './models/cell_collective/'
    OutFileName = './data/cell_collective/converged_average_node_values.csv'

if IMPORT_RULES_FROM_FILES:
    from os import listdir
    
    sync_models = {}
    for fname in listdir(models_dir):
        with open(models_dir+fname) as rulefile:
            name = fname.split(".")[0]
            rules = rulefile.read()
            sync_models[name]=cw.Model(rules)
else:
    from cana.datasets.bio import load_all_cell_collective_models
    def cell_collective_models():
        return {BN.name:cw.Model(cw.conversions.network_rules_from_cana(BN)) 
                for BN in load_all_cell_collective_models()}
    sync_models = cell_collective_models()
    for name,model in sync_models.items():
        with open(models_dir+name+'.txt','w') as rulefile:
            rulefile.write(model.rules)

total_models = len(sync_models)
async_models = deepcopy(sync_models)

difficult_models = {
    'Arabidopsis thaliana Cell Cycle': (5000,5000),
    'Guard Cell Abscisic Acid Signaling': (5000, 5000),
    'Signal Transduction in Fibroblasts': (20000, 50000),
}

def simulate_models(models, sync=True ,W=2500):
    convergence_measures = {}
    convergence_measures_alt = {}
    for model_idx, (model_name, model) in enumerate(sorted(models.items())):
        model.n_walkers = W
        N = model.n_variables
        
        if model_name in difficult_models:
            timescale, T_burn = difficult_models[model_name]
        else:
            timescale = N + 1000
            T_burn = N*50 + 1000
        
        T_window = 5 * timescale        
        T = T_burn + T_window            
        
        model.n_time_steps = T

        print(f"Simulating Model {model_name} ({W=},{T=},{T_burn=},{T_window=},{N=}). . .")
        if sync:
            model.simulate_ensemble(T_window=T_window,
                                    averages_only=True,
                                    maskfunction=cw.update_schemes.synchronous,
                                    threads_per_block=(16, 16))
        else:
            model.simulate_ensemble(T_window=T_window,
                                    averages_only=True,
                                    maskfunction=cw.update_schemes.asynchronous,
                                    threads_per_block=(16, 16))
            
        tw1 = model.trajectories[0*timescale:2*timescale]
        tw2 = model.trajectories[1*timescale:3*timescale]
        tw3 = model.trajectories[2*timescale:4*timescale]
        tw4 = model.trajectories[3*timescale:]
        convergence_measures[model_name] = max([
            cp.max(cp.abs(tw1.mean(axis=0) - tw2.mean(axis=0))),    
            cp.max(cp.abs(tw1.mean(axis=0) - tw3.mean(axis=0))),
            cp.max(cp.abs(tw1.mean(axis=0) - tw4.mean(axis=0))),
            cp.max(cp.abs(tw2.mean(axis=0) - tw3.mean(axis=0))),
            cp.max(cp.abs(tw2.mean(axis=0) - tw4.mean(axis=0))),
            cp.max(cp.abs(tw3.mean(axis=0) - tw4.mean(axis=0))),
            ])
        convergence_measures_alt[model_name] = max([
            cp.sum(cp.abs(tw1.mean(axis=0) - tw2.mean(axis=0))),    
            cp.sum(cp.abs(tw1.mean(axis=0) - tw3.mean(axis=0))),
            cp.sum(cp.abs(tw1.mean(axis=0) - tw4.mean(axis=0))),
            cp.sum(cp.abs(tw2.mean(axis=0) - tw3.mean(axis=0))),
            cp.sum(cp.abs(tw2.mean(axis=0) - tw4.mean(axis=0))),
            cp.sum(cp.abs(tw3.mean(axis=0) - tw4.mean(axis=0))),
            ])
        
        print(f"Progress: {(model_idx+1)}/{total_models},\t maximum difference: {convergence_measures[model_name]},\t sum difference: {convergence_measures_alt[model_name]}")
    return convergence_measures,convergence_measures_alt

sync_convergence_measures,sync_alt=simulate_models(sync_models,sync=True)

async_convergence_measures,async_alt=simulate_models(async_models,sync=False)

print(f'max max sync: {max(sync_convergence_measures.values())}')
print(f'max sum sycn: {max(sync_alt.values())}')
print(f'max max async: {max(async_convergence_measures.values())}')
print(f'max sum async: {max(async_alt.values())}')

with open(OutFileName,'w') as f:
    for model_name, smodel, amodel in [(k,sync_models[k],async_models[k]) for k in sorted(async_models)]:
        straj = ','.join(map(lambda x: str(cp.round(x,3)),cp.mean(smodel.trajectories,axis=0)))
        atraj = ','.join(map(lambda x: str(cp.round(x,3)),cp.mean(amodel.trajectories,axis=0)))
        f.write(f'{model_name},{cp.round(sync_convergence_measures[model_name],3)},{cp.round(sync_alt[model_name],3)},synchronous,{straj}\n')
        f.write(f'{model_name},{cp.round(async_convergence_measures[model_name],3)},{cp.round(async_alt[model_name],3)},asynchronous,{atraj}\n')