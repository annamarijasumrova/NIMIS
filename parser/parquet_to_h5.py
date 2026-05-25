import pandas as pd
import re
import ast
import h5py
import os 
import numpy as np
from rdkit import Chem
from pathlib import Path

file_path = Path("PATH_TO_NMREXP_PARQUET")
output_dir = Path(".")
h5_file = output_dir / "nmrexp_cno_hal.h5"

df = pd.read_parquet(file_path)
def processed_parser(processed):
    values = re.findall(r'\((.*?)\)', processed)
    parsed_data=[]
    invalid=False
    for entry in values:
        parts =ast.literal_eval(entry)
        if len(parts) < 5:
            invalid = True
            break
        multiplicity= parts[0]
        couplings_list = []
        raw = parts[1]
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, str):
            raw = raw.strip()
            if raw == "":
                items = []
            else:
                items = [x.strip() for x in raw.strip("[]").split(",")]
        else:
            items = []

        for c in items:
            s = str(c).lower()
            s = s.replace("hz", "")
            s = s.replace("j", "")
            s = s.replace("=", "")
            s = s.replace(":", "")
            s = s.strip().strip("'").strip('"')

            if s == "":
                continue

            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if m is None:
                continue
            couplings_list.append(float(m.group(0)))

        couplings = np.array(couplings_list, dtype=np.float32)

        delta=(parts[3]+parts[4])/2
        
        if delta > 16:
            invalid = True
            break
        
        if parts[2] is None:
            invalid = True
            break

        nb=parts[2].strip('H')
        if not nb.isdigit():
            invalid=True
            break
        else:
            nb_atoms=int(nb)

        if invalid:
            return None
        
        parsed_data.append([delta, nb_atoms, multiplicity, couplings])

    df_molecule = pd.DataFrame(parsed_data, columns=["delta", "nb_atoms", "multiplicity", "couplings"])
    return df_molecule


with h5py.File(h5_file, 'w') as h5f:
    for i in range(len(df)):
        print(i)
        smiles = df['SMILES'].iloc[i]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print('Invalid SMILES, skipping', i)
            continue

        atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        allowed_atoms = {'C', 'N', 'O', 'F', 'Cl', 'Br', 'I'}
        if not atom_symbols.issubset(allowed_atoms):
            print('Elements outside C/N/O/halogens present, skipping', i, atom_symbols)
            continue

        processed= df['NMR_processed'].iloc[i]
        peaks_df=processed_parser(processed)

        if peaks_df is None or peaks_df.empty:
            print('Skipping', i)
            continue
        
        mol_id=f"mol_{i}"
        group = h5f.create_group(mol_id)
        group.attrs['smiles'] = smiles
        group.attrs['solvent'] = df["NMR_solvent"].iloc[i]
        group.attrs['frequency'] = df["NMR_frequency"].iloc[i]

        group.create_dataset("nb_atoms", data=peaks_df['nb_atoms'].values.astype(np.int32))
        group.create_dataset("peak_shift", data=peaks_df['delta'].values.astype(np.float32))
        group.create_dataset("peak_multiplicity", data=peaks_df['multiplicity'].values.astype('S'))
        dt = h5py.vlen_dtype(np.float32)
        group.create_dataset("peak_coupling", data=peaks_df['couplings'].values, dtype=dt)

print(f"File ready: {h5_file}")

