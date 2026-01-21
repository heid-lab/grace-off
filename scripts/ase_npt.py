from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.constraints import FixCom

import argparse
import os
import time


parser = argparse.ArgumentParser(
    description="Run an ASE simulation with MLPotential."
)
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
    "--layer", type=str, required=False, help="Number of layers for GRACE model."
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
temperature = 300  # K
timestep = 0.5 * units.fs  # fs
log_interval = 100

mol = read(f"../data/liquids/{sol}.pdb")

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

mol.set_pbc([True, True, True])
# mol.set_constraint(FixCom()) # remove center of mass motion

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
dyn = IsotropicMTKNPT(
    mol,
    timestep,
    temperature_K=temperature,
    pressure_au=1.01325 * units.bar,
    tdamp=100 * timestep,
    pdamp=1000 * timestep,
    logfile=f"{path}/{sol}_{temperature}_npt.log",
    trajectory=f"{path}/{sol}_{temperature}_npt.traj",
    loginterval=log_interval,
)

_perf = {
    "start_wall": None,
    "last_wall": None,
    "last_t_int": None,
}  # t_int = ASE internal time

dens_csv = open(f"{path}/{sol}_{temperature}_npt_density.csv", "w", buffering=1)
dens_csv.write("time_ps,volume_A3,density_g_cm3,ns_per_day\n")


def log_density_csv():
    t_int = dyn.get_time()  # internal units (Å·sqrt(u/eV))
    t_fs = t_int / units.fs  # → femtoseconds
    t_ps = t_fs * 1e-3  # → picoseconds
    t_ns = t_fs * 1e-6  # → nanoseconds

    at = dyn.atoms
    V = at.get_volume()
    rho = AMU_TO_G__A3_TO_CM3 * mass_amu / V

    # wall clock
    now = time.time()
    if _perf["start_wall"] is None:
        _perf.update(start_wall=now, last_wall=now, last_t_int=t_int)

    wall_days = (now - _perf["start_wall"]) / 86400.0
    ns_per_day_cum = (t_ns / wall_days) if wall_days > 0 else float("nan")

    dens_csv.write(f"{t_ps:.4f},{V:.3f},{rho:.5f},{ns_per_day_cum:.5f}\n")

    _perf["last_wall"] = now
    _perf["last_t_int"] = t_int


dyn.attach(log_density_csv, interval=log_interval)

n_steps = 2_200_000
dyn.run(n_steps)
