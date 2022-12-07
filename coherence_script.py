# %%
import cubewalkers as cw
import cupy as cp
from copy import deepcopy

# %%
USE_FUZZY_COHERENCE = False

# %%
IMPORT_RULES_FROM_FILES = True
cc_models_dir = './models/cell_collective/'
if IMPORT_RULES_FROM_FILES:
    from os import listdir

    sync_models = {}
    for fname in listdir(cc_models_dir):
        with open(cc_models_dir+fname) as rulefile:
            name = fname.strip('.txt')
            rules = rulefile.read()
            sync_models[name] = cw.Model(rules)
else:
    from cana.datasets.bio import load_all_cell_collective_models

    def cell_collective_models():
        return {BN.name: cw.Model(cw.conversions.network_rules_from_cana(BN))
                for BN in load_all_cell_collective_models()}
    sync_models = cell_collective_models()
    for name, model in sync_models.items():
        with open(cc_models_dir+name+'.txt', 'w') as rulefile:
            rulefile.write(model.rules)

total_models = len(sync_models)
async_models = deepcopy(sync_models)

# %%


def simulate_model_coherence(models, sync=True, W=2500, fuzzy=USE_FUZZY_COHERENCE):
    coherences = {}
    for model_idx, (model_name, model) in enumerate(models.items()):
        model.n_walkers = W
        N = model.n_variables
        timescale = 2*N
        T = N**2 + 5 * timescale
        T_window = 5 * timescale
        model.n_time_steps = T

        print(f"Simulating Model {model_name} ({W=},{T=},{N=}). . .")
        if sync:
            maskfunction = cw.update_schemes.synchronous
        else:
            maskfunction = cw.update_schemes.asynchronous

        coherences[model_name] = model.quasicoherence(T_sample=T_window,
                                                      fuzzy_coherence=fuzzy,
                                                      maskfunction=maskfunction,
                                                      threads_per_block=(16, 16))

        print(
            f"Progress: {(model_idx+1)}/{total_models},\t coherence: {coherences[model_name]}")
    return coherences


# %%
sync_coherences = simulate_model_coherence(
    sync_models, sync=True, fuzzy=USE_FUZZY_COHERENCE)

# %%
async_coherences = simulate_model_coherence(
    async_models, sync=False, fuzzy=USE_FUZZY_COHERENCE)

# %%
if USE_FUZZY_COHERENCE:
    with open('./data/quasicohrence_fuzzy.csv', 'w') as f:
        f.write('model name, synchronous FQC, asynchronous FQC\n')
        for model_name, smodel, amodel in [(k, sync_models[k], async_models[k]) for k in sorted(async_models)]:
            f.write(
                f'{model_name},{cp.round(sync_coherences[model_name],3)},{cp.round(async_coherences[model_name],3)}\n')
else:
    with open('./data/quasicohrence_nonfuzzy.csv', 'w') as f:
        f.write('model name, synchronous QC, asynchronous QC\n')
        for model_name, smodel, amodel in [(k, sync_models[k], async_models[k]) for k in sorted(async_models)]:
            f.write(
                f'{model_name},{cp.round(sync_coherences[model_name],3)},{cp.round(async_coherences[model_name],3)}\n')

# %%
