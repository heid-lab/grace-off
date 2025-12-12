import numpy as np
from numpy.typing import NDArray
from openff.units import unit, Quantity
from typing import cast, Callable
from collections import namedtuple
import pandas as pd
from openff.units import unit
import warnings
from ase.io import read

warnings.filterwarnings("ignore")


##################### DEFINE PROPERTY FUNCTIONS ####################################


Constants = namedtuple("Constants", ["GAS_CONSTANT", "BOLTZMANN_CONSTANT"])

CONSTANTS = Constants(
    GAS_CONSTANT=8.31446261815324 * unit.joule / unit.mole / unit.kelvin,
    BOLTZMANN_CONSTANT=1.380649e-23 * unit.joule / unit.kelvin,
)


def calc_heat_capacity_units(
    total_energy: NDArray[np.float64],
    number_particles: int,
    temp: float,
    molar_mass: float,
    printing: bool,
) -> Quantity:
    """
    Compute the heat capacity.

    C_p = Variance(Total energy) / (number_particles * temp^2 * gas constant)
    the return value is in units cal/mole/kelvin
    for a correct unit transformation, the molar mass is required
    """
    pot_var = total_energy.var() * (unit.kilojoule / unit.mole) ** 2
    temp = temp * unit.kelvin

    val = cast(Quantity, pot_var / number_particles / temp**2 / CONSTANTS.GAS_CONSTANT)
    val = val.to(unit.cal / unit.mole / unit.kelvin) / molar_mass
    if printing:
        print("heat capacity: ", val)
    return val


def calc_thermal_expansion(
    total_energy: NDArray[np.float64],
    volume: NDArray[np.float64],
    temp: float,
    printing: bool,
) -> Quantity:
    """
    Compute the coefficient of thermal expansion.

    alpha = Cov(energy, vol) / (vol * temp^2 * gas constant)
    the return value is in units 1/Kelvin
    """
    cov_en_vol = cast(
        Quantity,
        np.cov(total_energy, volume)[0][1] * (unit.nanometer**3) * unit.kJ / unit.mole,
    )

    T = temp * unit.kelvin
    volume = volume.mean() * (unit.nanometer**3)

    alpha = cov_en_vol / CONSTANTS.GAS_CONSTANT / T**2 / volume
    alpha_shift = cast(Quantity, alpha.to(1 / unit.kelvin))
    if printing:
        print("thermal expansion: ", alpha_shift)
    return alpha_shift


def calc_isothermal_compressibility(
    volume: NDArray[np.float64], temp: float, printing: bool
) -> Quantity:
    """
    Compute the isothermal compressibility.

    kappa = Variance(Box volume) / (k_B * temperature * volume)
    the return value is in units 1/bar
    """
    volume_var = volume.var() * (unit.nanometer**3) ** 2
    volume_mean = volume.mean() * (unit.nanometer**3)
    T = temp * unit.kelvin

    val = volume_var / CONSTANTS.BOLTZMANN_CONSTANT / T / volume_mean
    val = val.to(1 / unit.bar)
    if printing:
        print("thermal expansion: ", val)
    return val


def calc_heat_of_vaporization(
    pot_energy: NDArray[np.float64],
    pot_energy_mono: NDArray[np.float64],
    temp: float,
    box_count: int,
    printing: bool,
) -> Quantity:
    """
    Compute the heat of vaporization.

    Delta H_vap = mean_energy_gas - mean_energy_liquid + R*temperature
    the return value is in units kJ
    """
    pot_mean = pot_energy.mean() * unit.kilojoule / unit.mole / box_count
    pot_mono_mean = pot_energy_mono.mean() * unit.kilojoule / unit.mole
    # temp_mean = temp_traj.mean() * unit.kelvin
    T = temp * unit.kelvin
    val = pot_mono_mean - pot_mean + CONSTANTS.GAS_CONSTANT * T
    if printing:
        print("heat of vaporozation: ", val)
    return val


############## PRINT EXPERIMENTAL WATER DATA #############################
theory = "Experiment"
print(f"{'Property':35} {'Model':10} {'Value':>10} {'Unit':>25}")
print("-" * 85)
print(
    f"{'Heat capacity':35} {theory:<10} {round(1.00, 2):>10} {str(unit.cal / unit.mole / unit.kelvin):>25}"
)
print(
    f"{'Isothermal compressibility (*1e4)':35} {theory:<10} {round(0.45, 2):>10} {str(1 / unit.bar):>25}"
)
print(
    f"{'Thermal expansion (*1e2)':35} {theory:<10} {round(0.03, 2):>10} {str(1 / unit.kelvin):>25}"
)
print(
    f"{'Heat of vaporization':35} {theory:<10} {round(43.99, 2):>10} {str(unit.kilojoule / unit.mole):>25}"
)
print(
    f"{'Density':35} {theory:<10} {round(0.997, 3):>10} {str((unit.gram / unit.milliliter)):>25}"
)
print("-" * 85)


############## COMPUTE PROPERTIES FROM WATER DATA #############################

eV_to_kjmol = 1.602176634e-19 * 6.02214076e23 / 1000


skip_size = 0.2  # skip the first 20% of the trajectory

models = [
    "2l_water_grace_small_b_off_run1",
    "2l_water_grace_medium_b_off",
    "2l_water_grace_large_b_off_run1",
    "2l_benzene_grace_small_b_off_run1",
    "2l_benzene_grace_medium_b_off_run1",
    "2l_benzene_grace_large_b_off_run1",    
    "2l_hexane_grace_small_b_off_run1",
    "2l_hexane_grace_medium_b_off_run1",
    "2l_hexane_grace_large_b_off_run1",
    "2l_acetone_grace_small_b_off_run1",
    "2l_acetone_grace_medium_b_off_run1",
    "2l_acetone_grace_large_b_off_run1",
    "2l_methanol_grace_small_b_off_run1",
    "2l_methanol_grace_medium_b_off_run1",
    "2l_methanol_grace_large_b_off_run1",
    "2l_nma_grace_small_b_off_run1",
    "2l_nma_grace_medium_b_off_run1",
    "2l_nma_grace_large_b_off_run1",
]

# Experimental values for different solutes
experimental_values = {
    "water": {
        "heat_of_vaporization": 43.99,
        "heat_capacity": 1.00,
        "compressibility": 0.45,
        "thermal_expansion": 0.03,
        "density": 0.997
    },
    "benzene": {
        "heat_of_vaporization": 33.83,  # kJ/mol
        "heat_capacity": 0.42, # CHATGPT value: 1.74,  # cal/(mol·K)
        "compressibility": 0.97,  # 1e-4 / bar
        "thermal_expansion": 0.11, # CHATGPUT: 1.24,  # 1e-2 / K
        "density": 0.88  # g/mL
    },
    "methanol": {
        "heat_of_vaporization": 38.3, # CHATGPUT: 37.43,
        "heat_capacity": 0.61, #CHATGPT 1.93,
        "compressibility": 0.31, # 1.22,
        "thermal_expansion": 0.15, # 1.19,
        "density": 0.79
    },
    "acetone": {
        "heat_of_vaporization": 31.9, # 30.99,
        "heat_capacity": 0.52, # 2.17,
        "compressibility": 1.41, # 1.27,
        "thermal_expansion": 0.15, # 1.43,
        "density": 0.78
    },
    "hexane": {
        "heat_of_vaporization": 31.52,
        "heat_capacity": 0.54, # 2.26,
        "compressibility": 1.67,
        "thermal_expansion": 0.14, # 1.39,
        "density": 0.66
    },
    "NMA": {
        "heat_of_vaporization": 53.50,  # kJ/mol
        "heat_capacity": "Nan",
        "compressibility": "Nan",  
        "thermal_expansion": "Nan", 
        "density": 0.92  # g/mL
    },
}

# Store results for CSV
csv_results = []

for model in models:
    print(f"\nCOMPUTING CONDENSED PHASE PROPERTIES for {model}\n")
    
    if "grace" in model.split("_"):
        sol = model.split("_")[1]
        temp = 300
        if sol == "nma":
            temp = 373
        liquid = pd.read_csv(f"../output/{model}/{sol}_{temp}_npt.log", sep="\s+")
        vol_data = pd.read_csv(f"../output/{model}/{sol}_{temp}_npt_density.csv")
        gas_data = pd.read_csv(f"../output/{model}/gas_{sol}_{temp}.log", sep="\s+")
        density = vol_data["density_g_cm3"].to_numpy().mean()
        speed = vol_data["ns_per_day"].to_numpy().mean()
        gas = gas_data["Epot[eV]"].to_numpy()
        mol = read(f"../data/liquids/{sol}.pdb")

        residue_numbers = set(mol.get_array("residuenumbers"))
        box_count = len(residue_numbers)
        print(f"Number of molecules in the box: {box_count}")

        total_mass = mol.get_masses().sum()  # Total mass of all atoms
        molar_mass = total_mass / box_count  # Mass per molecule in amu (g/mol)
        print(f"Molar mass: {molar_mass:.2f} g/mol")

    else:
        sol = model.split("_")[0]
        liquid = pd.read_csv(f"../output/{model}/{sol}_300_npt.log", sep="\s+")
        vol_data = pd.read_csv(f"../output/{model}/{sol}_300_npt_density.csv")
        density = vol_data["density_g_cm3"].to_numpy().mean()
        speed = vol_data["ns_per_day"].to_numpy().mean()
        gas_data = pd.read_csv(f"../output/{model}/gas_{sol}_300.log", sep="\s+")
        gas = gas_data["Epot[eV]"].to_numpy()
        box_count = 572
        molar_mass = 18.015 * unit.gram / unit.mole



    en_tot = liquid["Etot[eV]"].to_numpy()
    en_pot = liquid["Epot[eV]"].to_numpy()
    temp = liquid["T[K]"].to_numpy()
    vol = vol_data["volume_A3"].to_numpy() * 10**-3

    total_energy_kj_mol = en_tot * eV_to_kjmol
    potential_energy_kj_mol = en_pot * eV_to_kjmol
    gas_kj_mol = gas * eV_to_kjmol  # * beta

    heat_capacity = calc_heat_capacity_units(
        total_energy_kj_mol[int(len(total_energy_kj_mol) * skip_size) :],
        box_count,
        temp[int(len(temp) * skip_size) :].mean(),
        molar_mass,
        False,
    )
    index = min(len(total_energy_kj_mol), len(vol))
    thermal_expansion = calc_thermal_expansion(
        total_energy_kj_mol[int(len(total_energy_kj_mol) * skip_size) : index],
        vol[int(len(vol) * skip_size) : index],
        temp[int(len(temp) * skip_size) : index].mean(),
        False,
    )
    iso_comp = calc_isothermal_compressibility(
        vol[int(len(vol) * skip_size) :],
        temp[int(len(temp) * skip_size) :].mean(),
        False,
    )
    hov = calc_heat_of_vaporization(
        potential_energy_kj_mol[int(len(potential_energy_kj_mol) * skip_size) :],
        gas_kj_mol[int(len(gas_kj_mol) * skip_size) :],
        temp[int(len(temp) * skip_size) :].mean(),
        box_count,
        False,
    )

    density = vol_data["density_g_cm3"][int(len(vol_data["density_g_cm3"])*skip_size)].mean()
    speed = vol_data["ns_per_day"][int(len(vol) * skip_size) :].mean()

    # Determine readable model name
    if "grace" in model.lower():
        if "small" in model.lower():
            model_name = "GRACE (2L-S)"
        elif "medium" in model.lower():
            model_name = "GRACE (2L-M)"
        elif "large" in model.lower():
            model_name = "GRACE (2L-L)"
        else:
            model_name = "GRACE"
    elif "mace" in model.lower():
        if "small" in model.lower():
            model_name = "MACE (S) - OpenMM"
        elif "medium" in model.lower():
            model_name = "MACE (M) - OpenMM"
        elif "large" in model.lower():
            model_name = "MACE (L) - OpenMM"
        else:
            model_name = "MACE - OpenMM"
    else:
        model_name = model
    
    # Store results for CSV
    csv_results.append({
        "solute": sol.capitalize(),
        "model": model_name,
        "heat_of_vaporization": round(hov.magnitude, 2),
        "heat_capacity": round(heat_capacity.magnitude, 2),
        "compressibility": round(iso_comp.magnitude * 1e4, 2),
        "thermal_expansion": round(thermal_expansion.magnitude * 1e2, 2),
        "density": round(density, 3)
    })

    # Print results for this model
    print(f"\n{'Property':35} {'Model':20} {'Value':>12} {'Unit':>25}")
    print("-" * 95)
    print(
        f"{'Heat of vaporization':35} {model_name:<20} {round(hov.magnitude, 2):>12} {str(unit.kilojoule / unit.mole):>25}"
    )
    print(
        f"{'Heat capacity':35} {model_name:<20} {round(heat_capacity.magnitude, 2):>12} {str(unit.cal / unit.mole / unit.kelvin):>25}"
    )
    print(
        f"{'Isothermal compressibility (*1e4)':35} {model_name:<20} {round(iso_comp.magnitude*1e4, 2):>12} {str(1 / unit.bar):>25}"
    )
    print(
        f"{'Thermal expansion (*1e2)':35} {model_name:<20} {round(thermal_expansion.magnitude*1e2, 2):>12} {str(1 / unit.kelvin):>25}"
    )
    print(
        f"{'Density':35} {model_name:<20} {round(density, 3):>12} {str((unit.gram / unit.milliliter)):>25}"
    )
    print(
        f"{'Speed':35} {model_name:<20} {round(speed, 2):>12} {str(unit.nanoseconds / unit.day):>25}"
    )
    print("-" * 95)

# Add experimental values to results
for solute_name, exp_vals in experimental_values.items():
    csv_results.append({
        "solute": solute_name.capitalize(),
        "model": "experiment",
        "heat_of_vaporization": exp_vals["heat_of_vaporization"],
        "heat_capacity": exp_vals["heat_capacity"],
        "compressibility": exp_vals["compressibility"],
        "thermal_expansion": exp_vals["thermal_expansion"],
        "density": exp_vals["density"]
    })

# Add MACE model results
mace_results = [
    {"solute": "NMA", "model": "MACE (S) - OpenMM", "heat_of_vaporization": 58.83, "heat_capacity": 1.27, "compressibility": 0.48, "thermal_expansion": 0.11, "density": 1.07},
    {"solute": "NMA", "model": "MACE (M) - OpenMM", "heat_of_vaporization": 46.82, "heat_capacity": 1.11, "compressibility": 1.87, "thermal_expansion": 0.14, "density": 0.87},
    {"solute": "Water", "model": "MACE (S) - OpenMM", "heat_of_vaporization": 47.8, "heat_capacity": 1.71, "compressibility": 0.29, "thermal_expansion": 0.1, "density": 1.12},
    {"solute": "Water", "model": "MACE (M) - OpenMM", "heat_of_vaporization": 48.9, "heat_capacity": 1.57, "compressibility": 0.23, "thermal_expansion": 0.06, "density": 1.18},
    {"solute": "Methanol", "model": "MACE (S) - OpenMM", "heat_of_vaporization": 36.29, "heat_capacity": 1.43, "compressibility": 1.52, "thermal_expansion": 0.21, "density": 0.91},
    {"solute": "Methanol", "model": "MACE (M) - OpenMM", "heat_of_vaporization": 34.98, "heat_capacity": 1.4, "compressibility": 2.3, "thermal_expansion": 0.2, "density": 0.81},
    {"solute": "Acetone", "model": "MACE (S) - OpenMM", "heat_of_vaporization": 29.93, "heat_capacity": 1.25, "compressibility": 1.24, "thermal_expansion": 0.17, "density": 0.88},
    {"solute": "Acetone", "model": "MACE (M) - OpenMM", "heat_of_vaporization": 20.05, "heat_capacity": 1.23, "compressibility": 29.09, "thermal_expansion": 1.1, "density": 0.64},
    {"solute": "Benzene", "model": "MACE (S) - OpenMM", "heat_of_vaporization": 36.19, "heat_capacity": 1.18, "compressibility": 0.72, "thermal_expansion": 0.15, "density": 1.02},
    {"solute": "Benzene", "model": "MACE (M) - OpenMM", "heat_of_vaporization": 28.11, "heat_capacity": 1.24, "compressibility": 1.97, "thermal_expansion": 0.27, "density": 0.89},
    {"solute": "Hexane", "model": "MACE (S) - OpenMM", "heat_of_vaporization": 33.1, "heat_capacity": 2.19, "compressibility": 3.79, "thermal_expansion": 0.53, "density": 0.79},
    {"solute": "Hexane", "model": "MACE (M) - OpenMM", "heat_of_vaporization": 15.77, "heat_capacity": 2.12, "compressibility": 327.35, "thermal_expansion": 3.44, "density": 0.39},
]

csv_results.extend(mace_results)

# Save results to CSV
df_results = pd.DataFrame(csv_results)
df_results.to_csv("model_properties.csv", index=False)
print(f"\nResults saved to model_properties.csv")
