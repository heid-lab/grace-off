from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.langevin import Langevin
import numpy as np
from ase.md.npt import NPT
import sys
import argparse
import os

# from tensorpotential.calculator import TPCalculator

parser = argparse.ArgumentParser(
    description="Run an ase simulation with MLPotential."
)
parser.add_argument(
    "--sol", type=str, required=True, help="Solvent box."
)


args = parser.parse_args()


sol = args.sol # wat or moh

path = f"../data/mace/traj_{sol}"

os.makedirs(path, exist_ok=True)

# model_path = "/share/theochem/johannes.karwounopoulos/4d_test/foundation_models"
model_path = f"../models/MACE-OFF23_small.model"
# Constants for NPT dynamics
temperature = 300  # K
timestep = 0.5 * units.fs  # fs
ttime = 25 * units.fs
ptime=75 * units.fs
B_water = 2.2 * units.GPa  # ≃ 0.0137 eV/Å³
log_interval = 100


# for model in ["MACE-OFF24_medium.model", "MACE-OFF23_large.model","MACE-OFF23_small.model"]:
# for model in ["1l_c_all_medium.model"]:

# for temperature in [300, 315, 330]: #270, 285, 

temperature = 300
# name = model.split(".")[0].lower()
if sol == "wat":
    mol = read("../data/wat_mono.pdb")
else:
    mol = read(f"../data/{sol}_mace_npt_equil.pdb")
# mol.calc = TPCalculator(model=f"{model_path}", device="cuda")
mol.calc = MACECalculator(model_paths=model_path, device="cuda")
mol.set_pbc([False, False, False])

# Precompute total mass (amu) once
mass_amu = mol.get_masses().sum()
AMU_TO_G__A3_TO_CM3 = 1.66053906660  # g/cm^3 = 1.6605 * amu / Å^3

# set initial velocities
MaxwellBoltzmannDistribution(mol, temperature_K=temperature)
E_initial = mol.get_potential_energy()
print(f"Initial NPT energy: {E_initial:.6f} eV")

# short Geometry optimization
opt = BFGS(mol)
opt.run(steps=10)


dyn = Langevin(
    mol,
    timestep,
    temperature_K=temperature,
    friction=0.002,  # Damping factor (in fs⁻¹) – adjust as needed
    logfile=f"{path}/gas_{sol}_{temperature}.log",
    trajectory=f"{path}/gas_{sol}{temperature}.traj",
    loginterval=log_interval,
)

# instantiate NPT (drops in place of Langevin)
# dyn = NPT(
#     mol,
#     timestep,
#     temperature_K=temperature,
#     externalstress=1.0 * units.bar,
#     ttime=ttime,
#     pfactor=ptime**2 * B_water,
#     logfile=f"{path}/gas_{sol}_{temperature}.log",
#     trajectory=f"{path}/gas_{sol}{temperature}.traj",
#     loginterval=log_interval,
# )

# keep: logfile=log_file in NPT(...)
# dens_csv = open(f"{path}/gas_{sol}{temperature}_density.csv", "w", buffering=1)
# dens_csv.write("time_ps,volume_A3,density_g_cm3\n")

# def log_density_csv():
#     at = dyn.atoms
#     V = at.get_volume()
#     rho = AMU_TO_G__A3_TO_CM3 * mass_amu / V
#     t_ps = dyn.get_time() * 1e-3
#     dens_csv.write(f"{t_ps:.4f},{V:.3f},{rho:.5f}\n")

# dyn.attach(log_density_csv, interval=log_interval)

n_steps = 2_000_000
# dyn.set_fraction_traceless(0) # By setting this to zero, the volume may change but the shape may not (https://ase-lib.org/ase/md.html#ase.md.npt.NPT.set_fraction_traceless)
dyn.run(n_steps)



