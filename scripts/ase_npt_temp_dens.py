from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from ase import units
import numpy as np
from ase.md.npt import NPT
import sys

from tensorpotential.calculator import TPCalculator

# model_path = "/share/theochem/johannes.karwounopoulos/4d_test/foundation_models"
model_path = "/share/theochem/johannes.karwounopoulos/4d_test/train_grace/1_layer/c_all_medium_400_epochs/seed/1/saved_model/"
# Constants for NPT dynamics
temperature = 300  # K
timestep = 1 * units.fs  # fs
ttime = 25 * units.fs
ptime=75 * units.fs
B_water = 2.2 * units.GPa  # ≃ 0.0137 eV/Å³
log_interval = 100


# for model in ["MACE-OFF24_medium.model", "MACE-OFF23_large.model","MACE-OFF23_small.model"]:
for model in ["1l_c_all_medium.model"]:

    for temperature in [300, 315, 330]: #270, 285, 

        name = model.split(".")[0].lower()
        mol = read("mace_s_cptequil.pdb")
        mol.calc = TPCalculator(model=f"{model_path}", device="cuda")
        # mol.calc = MACECalculator(model_paths=f"{model_path}/{model}", device="cuda")
        mol.set_pbc([True, True, True])

        # Precompute total mass (amu) once
        mass_amu = mol.get_masses().sum()
        AMU_TO_G__A3_TO_CM3 = 1.66053906660  # g/cm^3 = 1.6605 * amu / Å^3

        # short Geometry optimization
        opt = BFGS(mol)
        opt.run(steps=10)

        # instantiate NPT (drops in place of Langevin)
        dyn = NPT(
            mol,
            timestep,
            temperature_K=temperature,
            externalstress=1.0 * units.bar,
            ttime=ttime,
            pfactor=ptime**2 * B_water,
            logfile=f"{name}_{temperature}.log",
            trajectory=f"{name}_{temperature}.traj",
            loginterval=log_interval,
        )

        # keep: logfile=log_file in NPT(...)
        dens_csv = open(f"{name}_{temperature}_density.csv", "w", buffering=1)
        dens_csv.write("time_ps,volume_A3,density_g_cm3\n")

        def log_density_csv():
            at = dyn.atoms
            V = at.get_volume()
            rho = AMU_TO_G__A3_TO_CM3 * mass_amu / V
            t_ps = dyn.get_time() * 1e-3
            dens_csv.write(f"{t_ps:.3f},{V:.3f},{rho:.5f}\n")

        dyn.attach(log_density_csv, interval=log_interval)

        n_steps = 20_000
        dyn.set_fraction_traceless(0) # By setting this to zero, the volume may change but the shape may not (https://ase-lib.org/ase/md.html#ase.md.npt.NPT.set_fraction_traceless)
        dyn.run(n_steps)



