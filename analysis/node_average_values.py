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

def simulate_models(models, sync=True ,W=2500):
    convergence_measures = {}
    for model_idx, (model_name, model) in enumerate(models.items()):
        model.n_walkers = W
        N = model.n_variables
        timescale=2*N
        T = N**2 + 5 * timescale
        T_window = 5 * timescale
        model.n_time_steps = T

        print(f"Simulating Model {model_name} ({W=},{T=},{N=}). . .")
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
        print(f"Progress: {(model_idx+1)}/{total_models},\t maximum difference: {convergence_measures[model_name]}")
    return convergence_measures

sync_convergence_measures=simulate_models(sync_models,sync=True)

async_convergence_measures=simulate_models(async_models,sync=False)

print(f'{max(sync_convergence_measures.values())}')
print(f'{max(async_convergence_measures.values())}')

with open(OutFileName,'w') as f:
    for model_name, smodel, amodel in [(k,sync_models[k],async_models[k]) for k in sorted(async_models)]:
        straj = ','.join(map(lambda x: str(cp.round(x,3)),cp.mean(smodel.trajectories,axis=0)))
        atraj = ','.join(map(lambda x: str(cp.round(x,3)),cp.mean(amodel.trajectories,axis=0)))
        f.write(f'{model_name},{cp.round(sync_convergence_measures[model_name],3)},synchronous,{straj}\n')
        f.write(f'{model_name},{cp.round(async_convergence_measures[model_name],3)},asynchronous,{atraj}\n')
