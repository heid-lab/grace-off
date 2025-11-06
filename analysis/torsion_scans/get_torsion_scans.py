import h5py
import matplotlib.pyplot as plt
import numpy as np
import random

from ase import Atoms, units
from ase.io import read, write, Trajectory
from ase import units
from mace.calculators import MACECalculator
from ase.data import chemical_symbols

from tensorpotential.calculator import TPCalculator
from tensorpotential.calculator.foundation_models import grace_fm

# from rdkit import Chem
import os
import math

from tqdm import tqdm
import sys
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase import units

import pandas as pd



def get_model_path():
    return "../../models"

def get_data_tnet_path():
    return "../../data/TNet500-SPICE.hdf5"


############################ compute grace scans #################################################
def get_area_error_grace_2l(small: bool, medium: bool, large: bool):
    """
        Compute AUC error between DFT data and GRACE-OFF 2l models
        input variables small, medium, large indicate for which models the error analysis is done
    """

    model_path = get_model_path()
    data_tnet = h5py.File(get_data_tnet_path(), "r")
    if small:
        calc_grace_small = TPCalculator(
            model=f"{model_path}/2l/b_off_small/seed/1/saved_model/"
        )
    if medium:
        calc_grace_medium = TPCalculator(
            model=f"{model_path}/2l/b_off_medium/seed/1/saved_model/"
        )
    if large:
        calc_grace_large = TPCalculator(
            model=f"{model_path}/2l/b_off_large/seed/1/saved_model/"
        )

    error_trapez_grace_small = []
    error_trapez_grace_medium = []
    error_trapez_grace_large = []

    # tqdm über Moleküle
    for smiles in tqdm(data_tnet.keys(), desc="Get error AUC for GRACE 2l models...", unit="mol", ncols=80):
        Z = data_tnet[smiles]["atomic_numbers"][:]
        symbols = [chemical_symbols[int(z)] for z in Z]

        torsion_energies_small = []
        torsion_energies_medium = []
        torsion_energies_large = []

        # DFT-Referenzkurve
        dft_energy = np.asarray(data_tnet[smiles]["dft total energy"][:]) * 27.211386245988

        # Schleife über Konformationen
        for conf in data_tnet[smiles]["conformations"]:
            pos = conf[:] * units.Bohr
            mol = Atoms(symbols=symbols, positions=pos)

            if small:
                mol.calc = calc_grace_small
                torsion_energies_small.append(mol.get_potential_energy())

            if medium:
                mol.calc = calc_grace_medium
                torsion_energies_medium.append(mol.get_potential_energy())

            if large:
                mol.calc = calc_grace_large
                torsion_energies_large.append(mol.get_potential_energy())

        # Nach der Konformationsschleife: x-Achse erst jetzt definieren
        x = np.arange(len(dft_energy))

        if small and torsion_energies_small:
            grace_small_error = (np.array(torsion_energies_small) - np.min(torsion_energies_small)) - (dft_energy - np.min(dft_energy))
            area_small = np.trapezoid(np.abs(grace_small_error), x)
            error_trapez_grace_small.append(area_small)

        if medium and torsion_energies_medium:
            grace_medium_error = (np.array(torsion_energies_medium) - np.min(torsion_energies_medium)) - (dft_energy - np.min(dft_energy))
            area_medium = np.trapezoid(np.abs(grace_medium_error), x)
            error_trapez_grace_medium.append(area_medium)

        if large and torsion_energies_large:
            grace_large_error = (np.array(torsion_energies_large) - np.min(torsion_energies_large)) - (dft_energy - np.min(dft_energy))
            area_large = np.trapezoid(np.abs(grace_large_error), x)
            error_trapez_grace_large.append(area_large)

    return error_trapez_grace_small, error_trapez_grace_medium, error_trapez_grace_large






def get_area_error_grace_1l(small: bool, medium: bool, large: bool):
    """
        Compute AUC error between DFT data and GRACE-OFF 1l models
        input variables small, medium, large indicate for which models the error analysis is done
    """
    

    model_path = get_model_path()
    data_tnet = h5py.File(get_data_tnet_path(), "r")
    if small:
        calc_grace_small = TPCalculator(
            model=f"{model_path}/1l/b_off_small/seed/1/saved_model/"
        )
    if medium:
        calc_grace_medium = TPCalculator(
            model=f"{model_path}/1l/b_off_medium/seed/1/saved_model/"
        )
    if large:
        calc_grace_large = TPCalculator(
            model=f"{model_path}/1l/b_off_large/seed/1/saved_model/"
        )

    error_trapez_grace_small = []
    error_trapez_grace_medium = []
    error_trapez_grace_large = []

    # tqdm über Moleküle
    for smiles in tqdm(data_tnet.keys(), desc="Get error AUC for GRACE 1l models...", unit="mol", ncols=80):
        Z = data_tnet[smiles]["atomic_numbers"][:]
        symbols = [chemical_symbols[int(z)] for z in Z]

        torsion_energies_small = []
        torsion_energies_medium = []
        torsion_energies_large = []

        # DFT-Referenzkurve
        dft_energy = np.asarray(data_tnet[smiles]["dft total energy"][:]) * 27.211386245988

        # Schleife über Konformationen
        for conf in data_tnet[smiles]["conformations"]:
            pos = conf[:] * units.Bohr
            mol = Atoms(symbols=symbols, positions=pos)

            if small:
                mol.calc = calc_grace_small
                torsion_energies_small.append(mol.get_potential_energy())

            if medium:
                mol.calc = calc_grace_medium
                torsion_energies_medium.append(mol.get_potential_energy())

            if large:
                mol.calc = calc_grace_large
                torsion_energies_large.append(mol.get_potential_energy())

        # Nach der Konformationsschleife: x-Achse erst jetzt definieren
        x = np.arange(len(dft_energy))

        if small and torsion_energies_small:
            grace_small_error = (np.array(torsion_energies_small) - np.min(torsion_energies_small)) - (dft_energy - np.min(dft_energy))
            area_small = np.trapezoid(np.abs(grace_small_error), x)
            error_trapez_grace_small.append(area_small)

        if medium and torsion_energies_medium:
            grace_medium_error = (np.array(torsion_energies_medium) - np.min(torsion_energies_medium)) - (dft_energy - np.min(dft_energy))
            area_medium = np.trapezoid(np.abs(grace_medium_error), x)
            error_trapez_grace_medium.append(area_medium)

        if large and torsion_energies_large:
            grace_large_error = (np.array(torsion_energies_large) - np.min(torsion_energies_large)) - (dft_energy - np.min(dft_energy))
            area_large = np.trapezoid(np.abs(grace_large_error), x)
            error_trapez_grace_large.append(area_large)

    return error_trapez_grace_small, error_trapez_grace_medium, error_trapez_grace_large






############################ compute mace scans #################################################


def get_area_error_mace(small: bool, medium: bool, large: bool):
    """
        Compute AUC error between DFT data and MACE-OFF23 models
        input variables small, medium, large indicate for which models the error analysis is done
    """
    

    model_path = get_model_path()
    data_tnet = h5py.File(get_data_tnet_path(), "r")
    if small:
        calc_mace_large = MACECalculator(
            model_paths=f"{model_path}/MACE-OFF23_large.model", device="cuda"
        )
    if medium:
        calc_mace_medium = MACECalculator(
            model_paths=f"{model_path}/MACE-OFF24_medium.model", device="cuda"
        )
    if large:
        calc_mace_small = MACECalculator(
            model_paths=f"{model_path}/MACE-OFF23_small.model", device="cuda"
        )
    error_trapez_mace_small = []
    error_trapez_mace_medium = []
    error_trapez_mace_large = []

    # tqdm über Moleküle
    for smiles in tqdm(data_tnet.keys(), desc="Get error AUC for MACE models...", unit="mol", ncols=80):
        Z = data_tnet[smiles]["atomic_numbers"][:]
        symbols = [chemical_symbols[int(z)] for z in Z]

        torsion_energies_small = []
        torsion_energies_medium = []
        torsion_energies_large = []

        # DFT-Referenzkurve
        dft_energy = np.asarray(data_tnet[smiles]["dft total energy"][:]) * 27.211386245988

        # Schleife über Konformationen
        for conf in data_tnet[smiles]["conformations"]:
            pos = conf[:] * units.Bohr
            mol = Atoms(symbols=symbols, positions=pos)

            if small:
                mol.calc = calc_mace_small
                torsion_energies_small.append(mol.get_potential_energy())

            if medium:
                mol.calc = calc_mace_medium
                torsion_energies_medium.append(mol.get_potential_energy())

            if large:
                mol.calc = calc_mace_large
                torsion_energies_large.append(mol.get_potential_energy())

        # Nach der Konformationsschleife: x-Achse erst jetzt definieren
        x = np.arange(len(dft_energy))

        if small and torsion_energies_small:
            mace_small_error = (np.array(torsion_energies_small) - np.min(torsion_energies_small)) - (dft_energy - np.min(dft_energy))
            area_small = np.trapezoid(np.abs(mace_small_error), x)
            error_trapez_mace_small.append(area_small)

        if medium and torsion_energies_medium:
            mace_medium_error = (np.array(torsion_energies_medium) - np.min(torsion_energies_medium)) - (dft_energy - np.min(dft_energy))
            area_medium = np.trapezoid(np.abs(mace_medium_error), x)
            error_trapez_mace_medium.append(area_medium)

        if large and torsion_energies_large:
            mace_large_error = (np.array(torsion_energies_large) - np.min(torsion_energies_large)) - (dft_energy - np.min(dft_energy))
            area_large = np.trapezoid(np.abs(mace_large_error), x)
            error_trapez_mace_large.append(area_large)

    return error_trapez_mace_small, error_trapez_mace_medium, error_trapez_mace_large


############################ compute single torsion scan #################################################

def get_single_torsion_grace(index, small_1l: bool, medium_1l: bool, large_1l: bool, small_2l: bool, medium_2l: bool, large_2l: bool):
    """
        Compute a complete torsion scan for a single molecule for GRACE models
        index: the molecule from the dataset the torsion scan will be made for
        small_1l, medium_1l, large_1l, small_2l, medium_2l, large_2l bool variables -> make torsion scans for only those models
    """



    model_path = get_model_path()
    data_tnet = h5py.File(get_data_tnet_path(), "r")
    smiles = list(data_tnet.keys())[index]
    Z = data_tnet[smiles]["atomic_numbers"][:]
    symbols = [chemical_symbols[int(z)] for z in Z]

    torsion_energies_small_1l = []
    torsion_energies_medium_1l = []
    torsion_energies_large_1l = []
    torsion_energies_small_2l = []
    torsion_energies_medium_2l = []
    torsion_energies_large_2l = []

    for i, conf in enumerate(data_tnet[smiles]["conformations"]):
        pos = data_tnet[smiles]["conformations"][i][:] * units.Bohr
        mol = Atoms(symbols=symbols, positions=pos)
        if small_1l:
            calc_grace_small = TPCalculator(
                model=f"{model_path}/1l/b_off_small/seed/1/saved_model/"
            )
            mol.calc = calc_grace_small
            energy = mol.get_potential_energy()
            torsion_energies_small_1l.append(energy)
        if medium_1l:
            calc_grace_medium = TPCalculator(
                model=f"{model_path}/1l/b_off_medium/seed/1/saved_model/"
            )
            mol.calc = calc_grace_medium
            energy = mol.get_potential_energy()
            torsion_energies_medium_1l.append(energy)
        if large_1l:
            calc_grace_large = TPCalculator(
                model=f"{model_path}/1l/b_off_large/seed/1/saved_model/"
            )
            mol.calc = calc_grace_large
            energy = mol.get_potential_energy()
            torsion_energies_large_1l.append(energy)
        if small_2l:
            calc_grace_small = TPCalculator(
                model=f"{model_path}/2l/b_off_small/seed/1/saved_model/"
            )
            mol.calc = calc_grace_small
            energy = mol.get_potential_energy()
            torsion_energies_small_2l.append(energy)
        if medium_2l:
            calc_grace_medium = TPCalculator(
                model=f"{model_path}/2l/b_off_medium/seed/1/saved_model/"
            )
            mol.calc = calc_grace_medium
            energy = mol.get_potential_energy()
            torsion_energies_medium_2l.append(energy)
        if large_2l:
            calc_grace_large = TPCalculator(
                model=f"{model_path}/2l/b_off_large/seed/1/saved_model/"
            )
            mol.calc = calc_grace_large
            energy = mol.get_potential_energy()
            torsion_energies_large_2l.append(energy)

    return torsion_energies_small_1l, torsion_energies_medium_1l, torsion_energies_large_1l, torsion_energies_small_2l, torsion_energies_medium_2l, torsion_energies_large_2l



##################################### compute and save torsion scan errors ########################################################
def save_grace_errors_1l(small: bool, medium: bool, large: bool):
    error_trapez_grace_small, error_trapez_grace_medium, error_trapez_grace_large = get_area_error_grace_1l(small, medium, large)
    print("Print data to files...")
    grace_dict = dict()
    if small:
        grace_dict['small'] = error_trapez_grace_small
    if medium:
        grace_dict['medium'] = error_trapez_grace_medium
    if large:
        grace_dict['large'] = error_trapez_grace_large


    # in DataFrame umwandeln
    df_grace = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in grace_dict.items()]))

    # CSV speichern
    df_grace.to_csv("grace_errors_1l.csv", index=False)
    print("1l GRACE errors saved to grace_errors.csv")



def save_grace_errors_2l(small: bool, medium: bool, large: bool):
    error_trapez_grace_small, error_trapez_grace_medium, error_trapez_grace_large = get_area_error_grace_2l(small, medium, large)
    print("Print data to files...")
    grace_dict = dict()
    if small:
        grace_dict['small'] = error_trapez_grace_small
    if medium:
        grace_dict['medium'] = error_trapez_grace_medium
    if large:
        grace_dict['large'] = error_trapez_grace_large


    # in DataFrame umwandeln
    df_grace = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in grace_dict.items()]))

    # CSV speichern
    df_grace.to_csv("grace_errors_2l.csv", index=False)
    print("2l GRACE errors saved to grace_errors.csv")







def save_mace_errors(small: bool, medium: bool, large: bool):
    error_trapez_mace_small, error_trapez_mace_medium, error_trapez_mace_large = get_area_error_mace(small, medium, large)
    print("Print data to files...")
    mace_dict = dict()
    if small:
        mace_dict['small'] = error_trapez_mace_small
    if medium:
        mace_dict['medium'] = error_trapez_mace_medium
    if large:
        mace_dict['large'] = error_trapez_mace_large


    # in DataFrame umwandeln
    df_mace = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in mace_dict.items()]))

    # CSV speichern
    df_mace.to_csv("mace_errors.csv", index=False)
    print("MACE errors saved to mace_errors.csv")



def save_single_torsion_scan(index, small_1l: bool, medium_1l: bool, large_1l: bool, small_2l: bool, medium_2l: bool, large_2l: bool):
    torsion_energies_small_1l, torsion_energies_medium_1l, torsion_energies_large_1l, torsion_energies_small_2l, torsion_energies_medium_2l, torsion_energies_large_2l = get_single_torsion_grace(0, small_1l, medium_1l, large_1l, small_2l, medium_2l, large_2l)
    print("Print data to files...")
    single_torsion_dict = dict()
    if small_1l:
        single_torsion_dict['small_1l'] = torsion_energies_small_1l
    if medium_1l:
        single_torsion_dict['medium_1l'] = torsion_energies_medium_1l
    if large_1l:
        single_torsion_dict['large_1l'] = torsion_energies_large_1l
    if small_2l:
        single_torsion_dict['small_2l'] = torsion_energies_small_2l
    if medium_2l:
        single_torsion_dict['medium_2l'] = torsion_energies_medium_2l
    if large_2l:
        single_torsion_dict['large_2l'] = torsion_energies_large_2l


    # in DataFrame umwandeln
    df_torsion_energies_small = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in single_torsion_dict.items()]))

    # CSV speichern
    df_torsion_energies_small.to_csv("single_torsion.csv", index=False)
    print("Single GRACE torsion scan saved to single_torsion.csv")





if __name__ == "__main__":
    # Open progress log file
    with open("progress.log", "w") as log_file:
        # Redirect all print() output to the log file
        sys.stdout = log_file
        sys.stderr = log_file  # optional: also log errors/warnings

        # Define a version of tqdm that writes to the same log file
        def tqdm_log(*args, **kwargs):
            return tqdm(*args, file=log_file, dynamic_ncols=False, ascii=True, **kwargs)

        # Monkey-patch tqdm globally so all internal uses go to the log
        import builtins
        builtins.tqdm = tqdm_log

        save_grace_errors_1l(True, True, False)
        save_grace_errors_2l(True, True, False)
        save_mace_errors(True, True, True)
        save_single_torsion_scan(0, True, True, False, True, True, False)
