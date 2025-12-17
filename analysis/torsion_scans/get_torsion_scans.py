import h5py
import matplotlib.pyplot as plt
import numpy as np
import random

from ase import Atoms, units
from ase.io import read, write, Trajectory
from ase import units
# from mace.calculators import MACECalculator
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
import torch, gc
import pandas as pd



def get_model_path():
    return "../../models"

def get_data_tnet_path():
    return "../../data/TNet500-SPICE.hdf5"


############################ compute grace scans #################################################
# def get_barrier_height_error_grace_2l():

def get_grace_torsion_energies(model_type: str, layers: str, size: str): 

    data_tnet = h5py.File(get_data_tnet_path(), "r")
    smiles_list = sorted(data_tnet.keys())  # fixierte Reihenfolge
    model_path = get_model_path()
    
    calc_grace = TPCalculator(
        model=f"{model_path}/{layers}/{model_type}_{size}/seed/1/saved_model/"
    )
    
    all_torsion_energies = dict()
    
    for smiles in tqdm(smiles_list, desc=f"Get torsion scan for {model_type} {layers} {size}...", unit="mol"):
        Z = data_tnet[smiles]["atomic_numbers"][:]
        symbols = [chemical_symbols[int(z)] for z in Z]
        
        torsion_energies = []
        for conf in data_tnet[smiles]["conformations"]:
            pos = conf[:] * units.Bohr
            mol = Atoms(symbols=symbols, positions=pos)
            mol.calc = calc_grace
            torsion_energies.append(mol.get_potential_energy())
        
        all_torsion_energies[smiles] = torsion_energies
    
    return all_torsion_energies




def get_AUC_HB_grace(model_type: str, layers: str, size: str):
    """
        Compute AUC error between DFT data and GRACE models
        Compute barrier height (BH) errors
        input variables small, medium, large indicate for which models the error analysis is done
    """

 
    data_tnet = h5py.File(get_data_tnet_path(), "r")
    torsion_energies_grace = get_grace_torsion_energies(model_type, layers, size) 
    print(f"Torsion energies computed for {model_type} GRACE {layers} {size} model...")
    AUC_error_grace = []
    HB_error_grace = []

    for smiles, grace_energy in tqdm(torsion_energies_grace.items(), desc="Compute errors"):
        dft_energy = np.asarray(data_tnet[smiles]["dft total energy"][:]) * 27.211386245988
        x = np.arange(len(dft_energy))
        
        grace_dft_diff = (np.array(grace_energy) - np.min(grace_energy)) - (dft_energy - np.min(dft_energy))
        area = np.trapezoid(np.abs(grace_dft_diff), x)
        AUC_error_grace.append(area)
        
        barrier_dft = np.max(dft_energy) - np.min(dft_energy)
        barrier_grace = np.max(grace_energy) - np.min(grace_energy)
        HB_error_grace.append(barrier_grace - barrier_dft)
    

    print("Print data to files...")
    grace_dict = dict()

    grace_dict['AUC'] = AUC_error_grace
    grace_dict['HB'] = HB_error_grace


    # in DataFrame umwandeln
    df_grace = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in grace_dict.items()]))

    folder_path = "error_files"

    # Ordner erstellen, falls er noch nicht existiert
    os.makedirs(folder_path, exist_ok=True)
    # CSV speichern
    df_grace.to_csv(f"error_files/grace_errors_{model_type}_{layers}_{size}.csv", index=False)
    print(f"{layers} {model_type} GRACE {size} errors saved to error_files/grace_errors_{model_type}_{layers}_{size}.csv")











############################ compute mace scans #################################################

def get_mace_torsion_energies(size: str): 

    data_tnet = h5py.File(get_data_tnet_path(), "r")
    smiles_list = sorted(data_tnet.keys())  # fixierte Reihenfolge
    model_path = get_model_path()
    
    if size=='medium':
        model_name="MACE-OFF24_medium.model"
    else:
        model_name=f"MACE-OFF23_{size}.model"
    # model_paths=f"{model_path}/{model_name}"
    # print(model_paths)
    from mace.calculators import MACECalculator
    calc_mace = MACECalculator(
        model_paths=f"{model_path}/{model_name}", device="cuda"
    )
    
    all_torsion_energies = dict()
    
    for smiles in tqdm(smiles_list, desc=f"Get torsion scan for MACE {size} model...", unit="mol"):
        Z = data_tnet[smiles]["atomic_numbers"][:]
        symbols = [chemical_symbols[int(z)] for z in Z]
        
        torsion_energies = []
        for conf in data_tnet[smiles]["conformations"]:
            pos = conf[:] * units.Bohr
            mol = Atoms(symbols=symbols, positions=pos)
            mol.calc = calc_mace
            torsion_energies.append(mol.get_potential_energy())
        
        all_torsion_energies[smiles] = torsion_energies
    
    return all_torsion_energies

def get_AUC_HB_mace(size: str):
    """
        Compute AUC error between DFT data and MACE-OFF models
        Compute height barrier (HB) for MACE-OFF models
        input variables small, medium, large indicate for which models the error analysis is done
    """
    

    model_path = get_model_path()
    data_tnet = h5py.File(get_data_tnet_path(), "r")

    all_torsion_energies_mace = get_mace_torsion_energies(size)
    print(f"Torsion energies computed for MACE {size} model...")
    AUC_error_mace = []
    HB_error_mace = []

    for smiles, mace_energy in tqdm(all_torsion_energies_mace.items(), desc="Compute errors"):
        dft_energy = np.asarray(data_tnet[smiles]["dft total energy"][:]) * 27.211386245988
        x = np.arange(len(dft_energy))
        
        mace_dft_diff = (np.array(mace_energy) - np.min(mace_energy)) - (dft_energy - np.min(dft_energy))
        area = np.trapezoid(np.abs(mace_dft_diff), x)
        AUC_error_mace.append(area)
        
        barrier_dft = np.max(dft_energy) - np.min(dft_energy)
        barrier_mace = np.max(mace_energy) - np.min(mace_energy)
        HB_error_mace.append(barrier_mace - barrier_dft)
    

    print("Print data to files...")
    mace_dict = dict()

    mace_dict['AUC'] = AUC_error_mace
    mace_dict['HB'] = HB_error_mace


    # in DataFrame umwandeln
    df_mace = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in mace_dict.items()]))

    folder_path = "error_files"

    # Ordner erstellen, falls er noch nicht existiert
    os.makedirs(folder_path, exist_ok=True)
    # CSV speichern
    df_mace.to_csv(f"error_files/mace_errors_{size}.csv", index=False)
    print(f"MACE {size} errors saved to error_files/mace_errors_{size}.csv")


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


# def save_grace_errors(model_type: str, layers: str, size: str):
#     # error_trapez_grace_small, error_trapez_grace_medium, error_trapez_grace_large = get_area_error_grace(model_type, layers, small, medium, large)
#     AUC_error_grace, HB_error_grace = get_AUC_HB_grace(model_type, layers, size)
#     print("Print data to files...")
#     grace_dict = dict()

#     grace_dict['AUC'] = AUC_error_grace
#     grace_dict['HB'] = HB_error_grace


#     # in DataFrame umwandeln
#     df_grace = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in grace_dict.items()]))

#     # CSV speichern
#     df_grace.to_csv(f"grace_errors_{model_type}_{layers}_{size}.csv", index=False)
#     print(f"{layers} {model_type} GRACE {size} errors saved to grace_errors_{model_type}_{layers}_{size}.csv")







# def save_mace_errors(small: bool, medium: bool, large: bool):
#     error_trapez_mace_small, error_trapez_mace_medium, error_trapez_mace_large = get_area_error_mace(small, medium, large)
#     print("Print data to files...")
#     mace_dict = dict()
#     if small:
#         mace_dict['small'] = error_trapez_mace_small
#     if medium:
#         mace_dict['medium'] = error_trapez_mace_medium
#     if large:
#         mace_dict['large'] = error_trapez_mace_large


#     # in DataFrame umwandeln
#     df_mace = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in mace_dict.items()]))

#     # CSV speichern
#     df_mace.to_csv("mace_errors.csv", index=False)
#     print("MACE errors saved to mace_errors.csv")



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
        print('we made it here 1')
        # Redirect all print() output to the log file
        sys.stdout = log_file
        sys.stderr = log_file  # optional: also log errors/warnings

        # Define a version of tqdm that writes to the same log file
        def tqdm_log(*args, **kwargs):
            return tqdm(*args, file=log_file, dynamic_ncols=False, ascii=True, **kwargs)

        # Monkey-patch tqdm globally so all internal uses go to the log
        import builtins
        builtins.tqdm = tqdm_log

        # comp_models_grace = [
        #     # ['a_wpS', '1l', 'small'],
        #     # ['a_wpS', '1l', 'medium'],
        #     # ['a_wpS', '1l', 'large'],
        #     # ['a_wpS', '2l', 'small'],
        #     # ['a_wpS', '2l', 'medium'],
        #     # ['a_wpS', '2l', 'large'],
        #     ['b_off', '1l', 'small'],
        #     ['b_off', '1l', 'medium'],
        #     ['b_off', '1l', 'large'],
        #     ['b_off', '2l', 'small'],
        #     ['b_off', '2l', 'medium'],
        #     ['b_off', '2l', 'large'],
        #     ]
        # for model in comp_models_grace:
        #     get_AUC_HB_grace(model[0], model[1], model[2])

        # comp_models_mace = ['small', 'medium', 'large']
        # for model in comp_models_mace:
        #     get_AUC_HB_mace(model)
        

        # save_mace_errors(True, True, True) # small / medium /large
        save_single_torsion_scan(0, True, True, True, True, True, True) # mol_index / small_1l / medium_1l /large_1l / small_2l / medium_2l / large_2l
