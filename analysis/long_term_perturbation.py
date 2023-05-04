from io import StringIO
import cubewalkers as cw
import cupy as cp
from copy import deepcopy

IMPORT_RULES_FROM_FILES = True
CORRECTED_MODELS = True

if CORRECTED_MODELS:
    models_dir = './models/corrected_models/'
    IMPORT_RULES_FROM_FILES = True # The corrections are only available in files
    OutFileName = './data/corrected_models/{comb}.csv'
else:
    models_dir = './models/cell_collective/'
    OutFileName = './data/cell_collective/{comb}.csv'
    

DEBUG_USING_SHORT_TIME = False # IF TRUE, OUTPUTS WILL NOT HAVE TIME TO CONVERGE; ONLY MODIFY IT TO GENERATE RESULTS
GLOBAL_WALKER_COUNT = 2500
GLOBAL_TPB = (16,16)
COMBINATIONS_TO_SIMULATE = ['quasicoherence_nonfuzzy_withsource',
                            'quasicoherence_fuzzy_withsource',
                            'final_hamming_distance_withsource',
                            'quasicoherence_nonfuzzy_sourceless',
                            'quasicoherence_fuzzy_sourceless',
                            'final_hamming_distance_sourceless']

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

def get_everything(trajU, trajP, diff, T_sample):
    c = cw.simulation.source_quasicoherence(trajU, trajP, T_sample, fuzzy_coherence=False)
    fc = cw.simulation.source_quasicoherence(trajU, trajP, T_sample, fuzzy_coherence=True)
    fhd = cw.simulation.source_final_hamming_distance(diff, T_sample)
    return cp.array([c, fc, fhd])

def simulate_models(models, sync, difficult_models=None):
    
    if difficult_models is None:
        difficult_models = {}
    
    results = {}
    for comb in COMBINATIONS_TO_SIMULATE:
        results[comb] = {}

    W = GLOBAL_WALKER_COUNT
    for model_idx, (model_name, model) in enumerate(models.items()):
        model.n_walkers = W
        N = model.n_variables
        results[model_name] = {}
        if not DEBUG_USING_SHORT_TIME:
            if model_name in difficult_models:
                timescale, T_burn = difficult_models[model_name]
            else:
                timescale = N + 1000
                T_burn = N*50 + 1000
            T_sample = 5 * timescale        
            T = T_burn + T_sample 
        else:
            T = 10
            T_sample = 5
        
        model.n_time_steps = T

        print(f"Simulating Model {model_name} ({W=},{T=},{N=}). . .")
        if sync:
            maskfunction = cw.update_schemes.synchronous
        else:
            maskfunction = cw.update_schemes.asynchronous

        withsource = cp.array([0.0,0.0,0.0])
        sourceless = cp.array([0.0,0.0,0.0])
        
        n_core = 0
        n_source = 0
        for rule in StringIO(model.rules):
            name, func = map(lambda x: x.strip(), rule.split(','))
            
            source = model.vardict[name]
            trajU, trajP, diff = cw.simulation.simulate_perturbation(model.kernel,
                                                                source,
                                                                model.n_variables,
                                                                model.n_time_steps,
                                                                model.n_walkers,
                                                                T_sample=T_sample,
                                                                lookup_tables=model.lookup_tables,
                                                                maskfunction=maskfunction,
                                                                threads_per_block=GLOBAL_TPB)

            add = get_everything(trajU, trajP, diff, T_sample)

            if name == func:
                n_source += 1
                withsource += add
            else:
                n_core += 1
                withsource += add
                sourceless += add

        n_all = n_source + n_core
        
        withsource /= n_all
        sourceless /= n_core
        sourceless[1] = 1- (1-sourceless[1])*n_all/n_core # renormalization

        final = cp.append(withsource,sourceless)
        for i, comb in enumerate(COMBINATIONS_TO_SIMULATE):
            results[comb][model_name] = final[i]

        print(f"Progress: {(model_idx+1)}/{total_models}")
    return results

models = import_models(models_dir, IMPORT_RULES_FROM_FILES=IMPORT_RULES_FROM_FILES)
total_models = len(models)

difficult_models = {
        'Arabidopsis thaliana Cell Cycle': (5000,5000),
        'Guard Cell Abscisic Acid Signaling': (5000, 5000),
        'Signal Transduction in Fibroblasts': (20000, 50000),
    }
sync_results=simulate_models(models,True, difficult_models=difficult_models)
async_results=simulate_models(models,False, difficult_models=difficult_models)

headers = ['model name, SQC, AQC\n',
           'model name, SFQC, AFQC\n',
           'model name, SFHD, AHFD\n',
           'model name, SQCNS, AQCNS\n',
           'model name, SFQCNS, AFQCNS\n',
           'model name, SFHDNS, AHFDNS\n']

for i, comb in enumerate(COMBINATIONS_TO_SIMULATE):
    filestring = OutFileName.format(comb = comb)

    with open(filestring,'w') as f:
        f.write(headers[i])
        for model_name, model in [(k, models[k]) for k in sorted(models)]:
            f.write((
                f'{model_name},'
                f'{cp.round(sync_results[comb][model_name],6)},'
                f'{cp.round(async_results[comb][model_name],6)}\n'
                ))