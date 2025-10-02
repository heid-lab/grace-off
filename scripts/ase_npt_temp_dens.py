from pathlib import Path
from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.npt import NPT
import numpy as np
import argparse
import os
import time

# from mace.calculators import MACECalculator
from tensorpotential.calculator import TPCalculator

wall_start = time.time()

parser = argparse.ArgumentParser(
    description="Run an OpenMM simulation with MLPotential."
)
parser.add_argument(
    "--layers", type=str, required=True, help="Layers of grace model."
)
parser.add_argument(
    "--model", type=str, required=True, help="Name of grace-off model."
)
parser.add_argument("--model_path", type=Path, default=None,
               help="Path to model (defaults to ../models/{layers}/{model}/seed/1/saved_model).")

args = parser.parse_args()

layers = args.layers # 1l
model = args.model # a_wpS_small

path = f"../data/traj/{layers}_{model}"

os.makedirs(path, exist_ok=True)

if args.model_path is None:
    args.model_path =  f"../models/{layers}/{model}/seed/1/saved_model"

print(f"Using model at {args.model_path}")

# Constants for NPT dynamics
temperature = 300  # K
timestep = 1 * units.fs  # fs
ttime = 25 * units.fs
ptime=75 * units.fs
B_water = 2.2 * units.GPa  # ≃ 0.0137 eV/Å³
log_interval = 100
temperature = 300


mol = read("../data/mace_s_cptequil.pdb")
mol.calc = TPCalculator(model=f"{args.model_path}", device="cuda")
# mol.calc = MACECalculator(model_paths=f"{model_path}/{model}", device="cuda")
mol.set_pbc([True, True, True])

# Precompute total mass (amu) once
mass_amu = mol.get_masses().sum()
AMU_TO_G__A3_TO_CM3 = 1.66053906660  # g/cm^3 = 1.6605 * amu / Å^3

# set initial velocities
MaxwellBoltzmannDistribution(mol, temperature_K=temperature)
E_initial = mol.get_potential_energy()
print(f"Initial NPT energy: {E_initial:.6f} eV")

# short Geometry optimization
# opt = BFGS(mol)
# opt.run(steps=50)

# instantiate NPT (drops in place of Langevin)
dyn = NPT(
    mol,
    timestep,
    temperature_K=temperature,
    externalstress=1.0 * units.bar,
    ttime=ttime,
    pfactor=ptime**2 * B_water,
    logfile=f"{path}/wat_{temperature}.log",
    trajectory=f"{path}/wat_{temperature}.traj",
    loginterval=log_interval,
)

# keep: logfile=log_file in NPT(...)
dens_csv = open(f"{path}/wat_{temperature}_density.csv", "w", buffering=1)
dens_csv.write("time_ps,volume_A3,density_g_cm3,ns_per_day_cum,ns_per_day\n")

def log_density_csv():
    at = dyn.atoms
    V = at.get_volume()
    rho = AMU_TO_G__A3_TO_CM3 * mass_amu / V
    t_int = dyn.get_time()              # internal units (Å·sqrt(u/eV))
    t_fs = t_int / units.fs 
    t_ps = t_fs * 1e-3                  # ps
    t_ns = t_fs * 1e-6                  # ns
    # wall time -> days
    wall_elapsed_s = time.time() - wall_start
    wall_days = wall_elapsed_s / 86400.0
    ns_per_day_cum = (t_ns / wall_days) if wall_days > 0 else float("nan")

    dens_csv.write(f"{t_ps:.3f},{V:.3f},{rho:.5f},{ns_per_day_cum:.5f}\n")


dyn.attach(log_density_csv, interval=log_interval)

n_steps = 20_000_000
dyn.set_fraction_traceless(0) # By setting this to zero, the volume may change but the shape may not (https://ase-lib.org/ase/md.html#ase.md.npt.NPT.set_fraction_traceless)
dyn.run(n_steps)



