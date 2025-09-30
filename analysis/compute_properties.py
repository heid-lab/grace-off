import numpy as np
from numpy.typing import NDArray
from openff.units import unit, Quantity
from typing import cast, Callable
from collections import namedtuple
import pandas as pd
from openff.units import unit
import warnings
warnings.filterwarnings("ignore")




##################### DEFINE PROPERTY FUNCTIONS ####################################


Constants = namedtuple('Constants', ['GAS_CONSTANT', 'BOLTZMANN_CONSTANT'])

CONSTANTS = Constants(
    GAS_CONSTANT=8.31446261815324 * unit.joule / unit.mole / unit.kelvin,
    BOLTZMANN_CONSTANT=1.380649e-23 * unit.joule / unit.kelvin,
)


def calc_heat_capacity_units(
    total_energy: NDArray[np.float64],
    number_particles: int,
    temp: float,
    molar_mass: float,
    printing: bool
) -> Quantity:
    """
    Compute the heat capacity.

    C_p = Variance(Total energy) / (number_particles * temp^2 * gas constant)
    the return value is in units cal/mole/kelvin
    for a correct unit transformation, the molar mass is required
    """
    pot_var = total_energy.var() * (unit.kilojoule / unit.mole) ** 2
    temp = temp * unit.kelvin

    val = cast(Quantity,
               pot_var
               / number_particles
               / temp**2
               / CONSTANTS.GAS_CONSTANT
               )
    val = val.to(unit.cal / unit.mole / unit.kelvin) / molar_mass
    if printing:
        print("heat capacity: ", val)
    return val


def calc_thermal_expansion(
        total_energy: NDArray[np.float64],
        volume: NDArray[np.float64],
        temp: float,
        printing: bool
) -> Quantity:
    """
    Compute the coefficient of thermal expansion.

    alpha = Cov(energy, vol) / (vol * temp^2 * gas constant)
    the return value is in units 1/Kelvin
    """
    cov_en_vol = cast(Quantity,
                      np.cov(total_energy, volume)[0][1]
                      * (unit.nanometer**3)
                      * unit.kJ / unit.mole
                      )

    T = temp * unit.kelvin
    volume = volume.mean() * (unit.nanometer**3)

    alpha = cov_en_vol / CONSTANTS.GAS_CONSTANT / T**2 / volume
    alpha_shift = cast(Quantity, alpha.to(1 / unit.kelvin))
    if printing:
        print("thermal expansion: ", alpha_shift)
    return alpha_shift


def calc_isothermal_compressibility(
        volume: NDArray[np.float64],
        temp: float,
        printing: bool
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
    temp_traj: NDArray[np.float64],
    box_count: int,
    printing: bool
) -> Quantity:
    """
    Compute the heat of vaporization.

    Delta H_vap = mean_energy_gas - mean_energy_liquid + R*temperature
    the return value is in units kJ
    """
    pot_mean = pot_energy.mean() * unit.kilojoule / unit.mole / box_count
    pot_mono_mean = pot_energy_mono.mean() * unit.kilojoule / unit.mole
    temp_mean = temp_traj.mean() * unit.kelvin
    val = pot_mono_mean - pot_mean + CONSTANTS.GAS_CONSTANT * temp_mean
    if printing:
        print("heat of vaporozation: ", val)
    return val


############## PRINT EXPERIMENTAL WATER DATA #############################
theory = "Experiment"
print(f"{'Property':35} {'Model':10} {'Value':>10} {'Unit':>25}")
print("-" * 85)
print(f"{'Heat capacity':35} {theory:<10} {round(1.00, 2):>10} {str(unit.cal / unit.mole / unit.kelvin):>25}")
print(f"{'Isothermal compressibility (*1e4)':35} {theory:<10} {round(0.45, 2):>10} {str(1 / unit.bar):>25}")
print(f"{'Thermal expansion (*1e2)':35} {theory:<10} {round(0.03, 2):>10} {str(1 / unit.kelvin):>25}")
# print(f"{'Heat of vaporization':35} {theory:<10} {round(hov.magnitude, 2):>10} {str(hov.units):>25}")
print(f"{'Density':35} {theory:<10} {round(0.997, 3):>10} {str((unit.gram / unit.milliliter)):>25}")
print("-" * 85)


############## COMPUTE PROPERTIES FROM WATER DATA #############################




skip_size = 0.1  # skip the first 10% of the trajectory

models = ['2l_a_wpS_small', '1l_a_wpS_small']

for model in models:
    liquid = pd.read_csv(f'../data/traj/{model}/wat_300.log', sep="\s+")
    vol_data = pd.read_csv("../data/traj/2l_a_wpS_medium/wat_300_density.csv")
    density = vol_data['density_g_cm3'].to_numpy().mean()
    box_count = 572
    molar_mass = 18.015 * unit.gram / unit.mole



    print(f"\nCOMPUTING CONDENSED PHASE PROPERTIES for {model}\n")

    # gas = get_gas_traj(theory)

    # skip_part_gas = int(round(gas["Potential Energy (kJ/mole)"].count()*skip_size,0))
    # gas_cut = gas[skip_part_gas-1:-1] # skip the first 10%

    en_tot = liquid["Etot[eV]"].to_numpy()
    temp = liquid['T[K]'].to_numpy()
    vol = vol_data['volume_A3'].to_numpy()*10**-3

    k_B = 8.617333e-5  # Boltzmann constant in eV/K
    beta = 1.0 / (k_B * temp.mean())  # beta in eV^-1
    total_energy_kj_mol = en_tot * beta

    skip_part_liquid = int(round(liquid["Etot[eV]"].count()*skip_size,0))
    liquid_cut = liquid[skip_part_liquid-1:-1] # skip the first 10%

    heat_capacity = calc_heat_capacity_units(
        total_energy_kj_mol, 
        box_count, 
        temp.mean(), 
        molar_mass, 
        False
    )
    index = min(len(total_energy_kj_mol),len(vol))
    thermal_expansion = calc_thermal_expansion(
        total_energy_kj_mol[:index], 
        vol[:index], 
        temp[:index].mean(), 
        False
    )

    iso_comp = calc_isothermal_compressibility(
        vol, 
        temp.mean(), 
        False
    )
    # hov = calc_heat_of_vaporization (
    #     liquid_cut["Potential Energy (kJ/mole)"].to_numpy(), 
    #     gas_cut["Potential Energy (kJ/mole)"].to_numpy(), 
    #     liquid_cut["Temperature (K)"].to_numpy(), 
    #     box_count, 
    #     False
    # )

    density = vol_data['density_g_cm3'].mean()
    theory = model
    print(f"{'Property':35} {'Model':10} {'Value':>10} {'Unit':>25}")
    print("-" * 85)
    print(f"{'Heat capacity':35} {theory:<10} {round(heat_capacity.magnitude, 2):>10} {str(heat_capacity.units):>25}")
    print(f"{'Isothermal compressibility (*1e4)':35} {theory:<10} {round(iso_comp.magnitude*1e4, 2):>10} {str(iso_comp.units):>25}")
    print(f"{'Thermal expansion (*1e2)':35} {theory:<10} {round(thermal_expansion.magnitude*1e2, 2):>10} {str(thermal_expansion.units):>25}")
    # print(f"{'Heat of vaporization':35} {theory:<10} {round(hov.magnitude, 2):>10} {str(hov.units):>25}")
    print(f"{'Density':35} {theory:<10} {round(density, 2):>10} {str((unit.gram / unit.milliliter)):>25}")
    print("-" * 85)


