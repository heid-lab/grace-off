import re
import glob
import csv
import os
import pandas as pd
import warnings

from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.constraints import FixCom

import argparse
import os
import time

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

from mace.calculators import MACECalculator
from tensorpotential.calculator import TPCalculator

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(
    description="Run an ASE simulation with MLPotential."
)
parser.add_argument(
    "--model_type", type=str, required=True, help="Model type (grace, mace)."
)

parser.add_argument(
    "--model_size", type=str, required=True, help="Model size (small, medium, large)."
)

parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    help="Dataset for GRACE training (either a_wpS or b_off).",
)
parser.add_argument(
    "--layers", type=int, required=False, help="Number of layers for GRACE model."
)
parser.add_argument(
    "--default_dtype",
    type=str,
    required=True,
    help="Single or double precision for energies.",
)


args = parser.parse_args()

model_type = args.model_type # grace / mace
model_size = args.model_size # small / medium / large
dataset = args.dataset # a_wpS / b_off
layers = args.layers # 1 / 2
default_dtype = args.default_dtype # float32 / float64



# model_type = 'grace' #args.model_type
# model_size = 'medium' #args.model_size
# dataset = 'b_off'
# layers = '2'
# default_dtype = 'float64' #args.default_dtype



if model_type.upper() == "GRACE":
    if default_dtype.lower() == "float64":
        model_path = (
            f"../../models/{layers}l/{dataset}_{model_size}/seed/1/saved_model"
        )
    else:
        model_path = (
            f"../../models/{layers}l/{dataset}_{model_size}/seed/1/casted_model"
        )
    
    print("Selected the following GRACE model:", model_path)
elif model_type.upper() == "MACE":
    if model_size=='medium':
        model_name = f"{model_type.upper()}-OFF24"
    else:
        model_name = f"{model_type.upper()}-OFF23"
    model_path = f"../../models/{model_name}_{model_size}.model"


print("Selected the following GRACE model:", model_path)


path = f"output/"
os.makedirs(path, exist_ok=True)






temp=300
dt=0.0005



# mace_model = 'small23' # small23 / medium23 / medium24
if model_type=='grace':
    output_file=f'{path}/energies_{model_type}_{dataset}_{layers}l_{model_size}.csv'
else:
    output_file=f'{path}/energies_{model_type}_{model_size}.csv'


write_header = not os.path.exists(output_file)



# pdb_files = sorted(glob.glob('pdbFiles/water_dimer_shifted_*.pdb'))

pattern = re.compile(r"pdbFiles/water_dimer_shifted_(-?\d+\.\d+)\.pdb$")

files = glob.glob("pdbFiles/water_dimer_shifted_*.pdb")

# dictionary to store {value: filename}
shift_dict = {}

for f in files:
    match = pattern.search(f)
    if match:
        value = float(match.group(1))
        shift_dict[value] = f

# sort by the numeric value
sorted_shifts = dict(sorted(shift_dict.items()))




with open(output_file, mode='a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'distance', 'energy_kJ_per_mol'])
    if write_header:
        writer.writeheader()


    # Example: iterate through in order
    for shift, pdb_file in sorted_shifts.items():
        print(f"{shift:5.2f} -> {pdb_file}")


        mol = read(pdb_file)
        if model_type.upper() == 'GRACE':
            mol.calc = TPCalculator(
                model=f"{model_path}", device="cuda", float_dtype=default_dtype
            )
        else:
            mol.calc = MACECalculator(
               model_paths=model_path, device="cuda", default_dtype=default_dtype
            )

        E_initial = mol.get_potential_energy()

        print(E_initial)
        writer.writerow({
            'distance': shift,
            'energy_kJ_per_mol': E_initial, #,.value_in_unit(unit.kilojoule_per_mole),
            'filename': pdb_file
        })




