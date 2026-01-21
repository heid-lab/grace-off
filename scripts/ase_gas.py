from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.constraints import FixCom

import argparse
import os


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
parser.add_argument("--run", type=int, required=False, help="Run number.")

args = parser.parse_args()

model_type = args.model_type
model_size = args.model_size
default_dtype = args.default_dtype
sol = args.sol
run = args.run if args.run is not None else 1

if model_type.upper() == "GRACE":
    from tensorpotential.calculator import TPCalculator
    if default_dtype.lower() == "float64":
        model_path = (
            f"../models/{args.layer}/{args.dataset}_{model_size}/seed/1/saved_model"
        )
    else:
        model_path = (
            f"../models/{args.layer}/{args.dataset}_{model_size}/seed/1/casted_model"
        )
    path = f"../output/{args.layer}_{sol}_{model_type}_{model_size}_{args.dataset}_run{run}"
    print("Selected the following GRACE model:", model_path)
elif model_type.upper() == "MACE":
    from mace.calculators import MACECalculator
    if model_size == "medium":
        model_path = f"../models/{model_type.upper()}-OFF24_{model_size}.model"
    else:
        model_path = f"../models/{model_type.upper()}-OFF23_{model_size}.model"
    path = f"../output/{sol}_{model_type}_{model_size}_run{run}"
elif model_type.upper() == "UMA":
    model_path = "uma-s-1p1"  # pretrained UMA model identifier
    path = f"../output/{sol}_{model_type}_run{run}"

os.makedirs(path, exist_ok=True)

# Constants for NPT dynamics
timestep = 0.5 * units.fs  # fs
log_interval = 1
temperature = 300

# Load only one molecule from the solvent box
orig_mol = read(f"../data/liquids/{sol}.pdb")
residue_nums = orig_mol.arrays.get('residuenumbers', [1] * len(orig_mol))
mol = orig_mol[[i for i, r in enumerate(residue_nums) if r == residue_nums[0]]]
mol.set_cell([False, False, False]) 
mol.center()

if model_type.upper() == "GRACE":
    mol.calc = TPCalculator(
        model=f"{model_path}", device="cuda", float_dtype=default_dtype
    )
elif model_type.upper() == "MACE":
    mol.calc = MACECalculator(
        model_paths=model_path, device="cuda", default_dtype=default_dtype
    )
elif model_type.upper() == "UMA":
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    predictor = pretrained_mlip.load_predict_unit("../models/uma-s-1p1.pt", device="cuda")
    mol.calc = FAIRChemCalculator(predictor, task_name="omol")

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

n_steps = 2_200_000
dyn.run(n_steps)
