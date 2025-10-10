from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.constraints import FixCom

import argparse
import os

from mace.calculators import MACECalculator
from tensorpotential.calculator import TPCalculator

parser = argparse.ArgumentParser(
    description="Run an OpenMM simulation with MLPotential."
)
parser.add_argument(
    "--model_size", type=str, required=True, help="Model size (small, medium, large)."
)
parser.add_argument(
    "--model_type", type=str, required=True, help="Architecture (either GRACE or MACE)."
)
parser.add_argument(
    "--dataset", type=str, required=False, help="Dataset for GRACE training (either a_wpS or b_off)."
)
parser.add_argument(
    "--sol", type=str, required=True, help="Solvent box."
)

args = parser.parse_args()

model_type = args.model_type
model_size = args.model_size
sol = args.sol

if model_type.upper() == "GRACE":
    model_path = f"../models/2l/{args.dataset}_{model_size}/seed/1/saved_model"
    path = f"../output/{sol}_{model_type}_{model_size}_{args.dataset}"
elif model_type.upper() == "MACE":
    model_path = f"../models/{model_type.upper()}-OFF23_{model_size}.model"
    path = f"../output/{sol}_{model_type}_{model_size}"

os.makedirs(path, exist_ok=True)

# Constants for NVT dynamics
timestep = 0.5 * units.fs  # fs
log_interval = 100
temperature = 300

mol = read("../data/mace_nvt_equil.pdb")

if model_type.upper() == "GRACE":
    mol.calc = TPCalculator(model=f"{model_path}", device="cuda")
elif model_type.upper() == "MACE":
    mol.calc = MACECalculator(model_paths=model_path, device="cuda")


mol.set_pbc([True, True, True])
mol.set_constraint(FixCom()) # remove center of mass motion


# Precompute total mass (amu) once
mass_amu = mol.get_masses().sum()
AMU_TO_G__A3_TO_CM3 = 1.66053906660  # g/cm^3 = 1.6605 * amu / Å^3

# set initial velocities
MaxwellBoltzmannDistribution(mol, temperature_K=temperature)
E_initial = mol.get_potential_energy()
print(f"Initial NPT energy: {E_initial:.6f} eV")

# short Geometry optimization
opt = BFGS(mol)
opt.run(steps=2)

dyn = NoseHooverChainNVT(
    mol, 
    timestep,
    temperature_K=temperature,
    tdamp=100*units.fs,
    logfile=f"{path}/{sol}_{temperature}_nvt.log",
    trajectory=f"{path}/{sol}_{temperature}_nvt.traj",
    loginterval=log_interval,
)

dens_csv = open(f"{path}/{sol}_{temperature}_nvt_density.csv", "w", buffering=1)
dens_csv.write("time_ps,volume_A3,density_g_cm3\n")

def log_density_csv():
    at = dyn.atoms
    V = at.get_volume()
    rho = AMU_TO_G__A3_TO_CM3 * mass_amu / V
    t_ps = dyn.get_time() * 1e-3
    dens_csv.write(f"{t_ps:.4f},{V:.3f},{rho:.5f}\n")

dyn.attach(log_density_csv, interval=log_interval)

n_steps = 2_200_000
dyn.run(n_steps)