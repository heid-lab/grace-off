from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.npt import NPT
from ase.constraints import FixCom

import argparse
import os
import time

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

# Constants for NPT dynamics
temperature = 300  # K
timestep = 0.5 * units.fs  # fs
ttime = 25 * units.fs
ptime=75 * units.fs
B_water = 2.2 * units.GPa  # ≃ 0.0137 eV/Å³
log_interval = 100

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
opt.run(steps=50)

# NPT dynamics
dyn = NPT(
    mol,
    timestep,
    temperature_K=temperature,
    externalstress=1.0 * units.bar,
    ttime=ttime,
    pfactor=ptime**2 * B_water,
    logfile=f"{path}/{sol}_{temperature}_npt.log",
    trajectory=f"{path}/{sol}_{temperature}_npt.traj",
    loginterval=log_interval,
)

_perf = {"start_wall": None, "last_wall": None, "last_t_int": None}  # t_int = ASE internal time

dens_csv = open(f"{path}/{sol}_{temperature}_npt_density.csv", "w", buffering=1)
dens_csv.write("time_ps,volume_A3,density_g_cm3\n")

def log_density_csv():
    t_int = dyn.get_time()                  # internal units (Å·sqrt(u/eV))
    t_fs = t_int / units.fs                 # → femtoseconds
    t_ps = t_fs * 1e-3                      # → picoseconds
    t_ns = t_fs * 1e-6                      # → nanoseconds

    at = dyn.atoms
    V = at.get_volume()
    rho = AMU_TO_G__A3_TO_CM3 * mass_amu / V

    # wall clock
    now = time.time()
    if _perf["start_wall"] is None:
        _perf.update(start_wall=now, last_wall=now, last_t_int=t_int)

    wall_days = (now - _perf["start_wall"]) / 86400.0
    ns_per_day_cum = (t_ns / wall_days) if wall_days > 0 else float("nan")

    dt_ns = (t_int - _perf["last_t_int"]) / units.fs * 1e-6
    dday = (now - _perf["last_wall"]) / 86400.0
    ns_per_day_inst = (dt_ns / dday) if dday > 0 else float("nan")

    dens_csv.write(f"{t_ps:.4f},{V:.3f},{rho:.5f},{ns_per_day_cum:.5f},{ns_per_day_inst:.5f}\n")

    _perf["last_wall"] = now
    _perf["last_t_int"] = t_int

dyn.attach(log_density_csv, interval=log_interval)

n_steps = 2_200_000
dyn.set_fraction_traceless(0) # By setting this to zero, the volume may change but the shape may not (https://ase-lib.org/ase/md.html#ase.md.npt.NPT.set_fraction_traceless)
dyn.run(n_steps)



