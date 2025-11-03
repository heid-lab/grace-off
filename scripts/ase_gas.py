from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.constraints import FixCom

import argparse
import os

from mace.calculators import MACECalculator
from tensorpotential.calculator import TPCalculator

parser = argparse.ArgumentParser(description="Run an ASE simulation with MLPotential.")
parser.add_argument(
    "--model_size", type=str, required=True, help="Model size (small, medium, large)."
)
parser.add_argument(
    "--model_type", type=str, required=True, help="Architecture (either GRACE or MACE)."
)
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    help="Dataset for GRACE training (either a_wpS or b_off).",
)
parser.add_argument(
    "--layer", type=int, required=False, help="Number of layers for GRACE model."
)
parser.add_argument(
    "--default_dtype",
    type=str,
    required=True,
    help="Single or double precision for energies.",
)
parser.add_argument("--sol", type=str, required=True, help="Solvent box.")

args = parser.parse_args()

model_type = args.model_type
model_size = args.model_size
default_dtype = args.default_dtype
sol = args.sol

if model_type.upper() == "GRACE":
    if default_dtype.lower() == "float64":
        model_path = (
            f"../models/{args.layer}l/{args.dataset}_{model_size}/seed/1/saved_model"
        )
    else:
        model_path = (
            f"../models/{args.layer}l/{args.dataset}_{model_size}/seed/1/casted_model"
        )
    path = f"../output/{args.layer}l_{sol}_{model_type}_{model_size}_{args.dataset}"
    print("Selected the following GRACE model:", model_path)
elif model_type.upper() == "MACE":
    model_path = f"../models/{model_type.upper()}-OFF23_{model_size}.model"
    path = f"../output/{sol}_{model_type}_{model_size}"

os.makedirs(path, exist_ok=True)

# Constants for NPT dynamics
timestep = 1 * units.fs  # fs
log_interval = 1
temperature = 300

mol = read(f"../data/{sol}_mono.pdb")

if model_type.upper() == "GRACE":
    mol.calc = TPCalculator(
        model=f"{model_path}", device="cuda", float_dtype=default_dtype
    )
elif model_type.upper() == "MACE":
    mol.calc = MACECalculator(
        model_paths=model_path, device="cuda", default_dtype=default_dtype
    )

# mol.set_cell([100.0, 100.0, 100.0])
mol.set_pbc([False, False, False])
mol.set_constraint(FixCom())  # remove center of mass motion

# set initial velocities
MaxwellBoltzmannDistribution(mol, temperature_K=temperature)
E_initial = mol.get_potential_energy()
print(f"Initial energy: {E_initial:.6f} eV")

# short Geometry optimization
opt = BFGS(mol)
opt.run(steps=50)

dyn = NoseHooverChainNVT(
    mol,
    timestep,
    temperature_K=temperature,
    tdamp=100 * timestep,
    logfile=f"{path}/gas_{sol}_{temperature}.log",
    trajectory=f"{path}/gas_{sol}_{temperature}.traj",
    loginterval=log_interval,
)

n_steps = 1_200_000
dyn.run(n_steps)
