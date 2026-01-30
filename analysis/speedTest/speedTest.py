from ase.io import read
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.nose_hoover_chain import IsotropicMTKNPT

import os
import time
import argparse

args = argparse.ArgumentParser(description="Speed test for different ML potentials")
arch = args.add_argument(
    "--arch",
    type=str,
    default="grace",
    help="Model architecture to test: grace, mace, uma",
)
args = args.parse_args()

if args.arch == "mace":
    grids = {
        # "grace": {"dtypes": ["float32", "float64"], "sizes": ["small", "medium", "large"]},
        "mace": {
            "dtypes": ["float32", "float64"],
            "sizes": ["small", "medium", "large"],
        },
        # "uma":   {"dtypes": ["float32"],                 "sizes": ["small"]},
    }
elif args.arch == "uma":
    grids = {
        # "grace": {"dtypes": ["float32", "float64"], "sizes": ["small", "medium", "large"]},
        # "mace": {"dtypes": ["float32", "float64"], "sizes": ["small", "medium", "large"]},
        "uma": {"dtypes": ["float32"], "sizes": ["small"]},
    }
elif args.arch == "grace":
    grids = {
        "grace": {
            "dtypes": ["float32", "float64"],
            "sizes": ["small", "medium", "large"],
        },
        # "mace": {"dtypes": ["float32", "float64"], "sizes": ["small", "medium", "large"]},
        # "uma":   {"dtypes": ["float32"],                 "sizes": ["small"]},
    }


for model_type, g in grids.items():
    for default_dtype in g["dtypes"]:
        for model_size in g["sizes"]:
            print(f"Now using: {model_type}, {model_size}, {default_dtype}")

            layer = "2"

            # constants
            sol = "water"
            temperature = 300  # K
            timestep = 1 * units.fs  # fs

            if model_type.upper() == "GRACE":
                from tensorpotential.calculator import TPCalculator
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                gpu_name = tf.config.experimental.get_device_details(gpus[0]).get(
                    "device_name", ""
                )
                if default_dtype.lower() == "float64":
                    model_path = (
                        f"../models/{layer}l/b_off_{model_size}/seed/1/saved_model"
                    )
                else:
                    model_path = (
                        f"../models/{layer}l/b_off_{model_size}/seed/1/casted_model"
                    )
                print("Selected the following GRACE model:", model_path)
            elif model_type.upper() == "MACE":
                from mace.calculators import MACECalculator
                import torch

                gpu_name = torch.cuda.get_device_name()

                if model_size == "medium":
                    model_path = (
                        f"../models/{model_type.upper()}-OFF24_{model_size}.model"
                    )
                else:
                    model_path = (
                        f"../models/{model_type.upper()}-OFF23_{model_size}.model"
                    )
            elif model_type.upper() == "UMA":
                from fairchem.core import pretrained_mlip, FAIRChemCalculator
                import torch

                gpu_name = torch.cuda.get_device_name()

                model_path = "uma-s-1p1"  # pretrained UMA model identifier

            mol = read(f"../../data/liquids/{sol}.pdb")

            if model_type.upper() == "GRACE":
                mol.calc = TPCalculator(
                    model=f"{model_path}", device="cuda", float_dtype=default_dtype
                )
            elif model_type.upper() == "MACE":
                mol.calc = MACECalculator(
                    model_paths=model_path, device="cuda", default_dtype=default_dtype
                )
            elif model_type.upper() == "UMA":
                predictor = pretrained_mlip.load_predict_unit(
                    "../models/uma-s-1p1.pt", device="cuda"
                )
                mol.calc = FAIRChemCalculator(predictor, task_name="omol")

            mol.set_pbc([True, True, True])

            # set initial velocities
            MaxwellBoltzmannDistribution(mol, temperature_K=temperature)
            E_initial = mol.get_potential_energy()

            # short Geometry optimization
            opt = BFGS(mol)
            opt.run(steps=50)

            # NPT dynamics
            dyn = IsotropicMTKNPT(
                mol,
                timestep,
                temperature_K=temperature,
                pressure_au=1.01325 * units.bar,
                tdamp=100 * timestep,
                pdamp=1000 * timestep,
                logfile=None,
                trajectory=None,
            )

            # Short ramp-up run
            print("Starting a short ram-up process")
            dyn.run(500)

            start = time.time()

            n_steps = 1_000
            print(f"Starting the actual timinig run for {n_steps} steps")
            dyn.run(n_steps)
            end = time.time()

            elapsedTime = end - start
            steps_per_second = n_steps / elapsedTime
            ns_day = (steps_per_second * 86400) * 1e-6
            print(f"Simulation speed: {ns_day:.2f} ns/day on {gpu_name}")

            results_file = "speed_test_results.csv"

            # Check if file exists, if not create it with headers
            if not os.path.exists(results_file):
                with open(results_file, "w") as f:
                    f.write(
                        "gpu_name,model_type,model_size,layer,default_dtype,n_steps, elapsed_time_s,ns_per_day,steps_per_second\n"
                    )
                print(f"Created new results file: {results_file}")
            else:
                print(f"Appending to existing results file: {results_file}")

            # Append current run results
            with open(results_file, "a") as f:
                f.write(
                    f"{gpu_name},{model_type.upper()},{model_size},{layer},{default_dtype},{n_steps},{elapsedTime:.2f},{ns_day:.2f},{steps_per_second:.2f}\n"
                )
