# type: ignore

# use latest booleannet from https://github.com/ialbert/booleannet
# use latest CANA from https://github.com/CASCI-lab/CANA
import random
import re
import time
from multiprocessing import Pool
from os import listdir

import boolean2
import cana.boolean_network as bn
import cubewalkers as cw

cana_T = 2500
cana_W = 2500

cw_T = 2500
cw_W = 2500

bn_T = 500
bn_W = 500
cc_models_dir = "./models/corrected_models/"
output_file = "./data/corrected_models/simple_benchmarks.csv"

with open(output_file, "w") as f:
    f.write(
        (
            "model name,"
            "model size,"
            "cana timesteps,"
            "cana walkers,"
            "cana time,"
            "cubewalkers timesteps,"
            "cubewalkers walkers,"
            "cubewalkers time,"
            "boolean2 timesteps,"
            "boolean2 walkers,"
            "boolean2 time\n"
        )
    )
    for fname in listdir(cc_models_dir):
        with open(cc_models_dir + fname) as rulefile:
            name = fname.split(".")[0]
            rules = rulefile.read()
        rules = re.sub(r";", r"_", rules)
        rules = re.sub(r"\&\&", r"and", rules)
        rules = re.sub(r"\|\|", r"or", rules)
        rules = re.sub(r"\!", r"not ", rules)
        rules = re.sub(r",\s+", r"*=", rules)
        rules = re.sub(r"\*=1", r"*=True", rules)
        rules = re.sub(r"\*=0", r"*=False", rules)

        # boolean2 gets confused by the case-sensitive rules for this model
        if (
            fname
            == "Iron acquisition and oxidative stress response in aspergillus fumigatus.txt"
        ):
            rules = re.sub(r"sreA", r"lowerSreA", rules)
            rules = re.sub(r"hapX", r"lowerHapX", rules)
        print(f"simulating model: {fname.split('.')[0]} . . .")
        ti = time.perf_counter()
        model_cana = bn.BooleanNetwork.from_string_boolean(rules)

        def worker(_):
            initial = "".join(random.choices(["0", "1"], k=model_cana.Nnodes))
            model_cana.trajectory(initial, length=cana_T)

        with Pool() as p:
            p.map(worker, range(cana_W))
        # for _ in range(cana_W):
        #     model_cana.trajectory(initial, length=cana_T)
        cana_time = time.perf_counter() - ti

        ti = time.perf_counter()
        model_cw = cw.Model(rules, n_time_steps=cw_T, n_walkers=cw_W)
        model_cw.simulate_ensemble(threads_per_block=(16, 16))
        cw_time = time.perf_counter() - ti

        # boolean2 cannot run this model for some reason
        if fname == "Signaling in Macrophage Activation.txt":
            print(
                f"N={model_cw.n_variables}; cana: {cana_time:.2f} s, cw: {cw_time:.2f} s, bn: NaN"
            )
            bn_time = float("NaN")
        else:
            ti = time.perf_counter()
            model_bn2 = boolean2.Model(text=rules, mode="sync")
            model_bn2.initialize(missing=lambda x: random.choice([0, 1]))

            def worker(_):
                model_bn2.iterate(steps=bn_T)

            with Pool() as p:
                p.map(worker, range(bn_W))
            # for _ in range(bn_W):
            #     model_bn2.iterate(steps=bn_T)
            bn_time = time.perf_counter() - ti

        print(
            f"N={model_cw.n_variables}; cana: {cana_time:.2f} s, cw: {cw_time:.2f} s, bn: {bn_time:.2f} s"
        )
        f.write(
            (
                f"{fname.split('.')[0]},"
                f"{model_cw.n_variables},"
                f"{cana_T},"
                f"{cana_W},"
                f"{cana_time},"
                f"{cw_T},"
                f"{cw_W},"
                f"{cw_time},"
                f"{bn_T},"
                f"{bn_W},"
                f"{bn_time}\n"
            )
        )
