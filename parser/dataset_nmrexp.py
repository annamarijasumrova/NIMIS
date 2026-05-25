import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random

class NMRMixtureClassificationNoSolvent(Dataset):
    def __init__(self, h5_path, split = 'train'):

        self.file = h5py.File(h5_path, "r")
        self.split = split
        self.all_keys = list(self.file.keys())
        self.total_N = len(self.all_keys)
        
        if split == 'train':
            subset_size = int(self.total_N * 0.05)
            self.keys = random.sample(self.all_keys, subset_size)
        else:
            subset_size = int(self.total_N * 0.01)
            self.keys = random.sample(self.all_keys, subset_size)

        self.N = len(self.keys)

    def __len__(self):
        return self.N




    def generate_scales(self, n_components, min_val=0.15):
        while True:
            scales = np.random.dirichlet(alpha=[1.0] * n_components)
            if np.all(scales >= min_val):
                return scales

    def __getitem__(self, idx):
        while True:
            n_components = random.choice((2, 3, 4))
            comp_ids = random.sample(range(self.N), n_components)
            scales = self.generate_scales(n_components)
            # print('chosen number of components is:', n_components, 'comp_id is', comp_ids)
            comp_integrals, comp_peaks, comp_smiles = [], [], []
            valid =True

            num_j = 3
            comp_j_vals = []
            comp_j_mask = []

            for i, cid in enumerate(comp_ids):
                group = self.file[self.keys[cid]]
                smiles = group.attrs["smiles"]
                nb_atoms = group["nb_atoms"][...].astype(np.float32)
                peaks = group["peak_shift"][...].astype(np.float32)
                j_raw = group["peak_coupling"][...]
                if peaks.size==0:
                    valid = False #breaks inner loop
                    break

                assert len(j_raw) == peaks.shape[0], "j_raw and peaks length mismatch"

                n_peaks_comp = peaks.shape[0]
                j_vals_comp = np.zeros((n_peaks_comp, num_j), dtype=np.float32)
                j_mask_comp = np.zeros((n_peaks_comp, num_j), dtype=bool)

                for p in range(n_peaks_comp):
                    js = np.array(j_raw[p], dtype=np.float32) 

                    if js.size == 0:
                        continue

                    js = js[~np.isnan(js)]
                    if js.size == 0:
                        continue

                    js_sorted = np.sort(js)[::-1]

                    k = min(num_j, js_sorted.size)
                    j_vals_comp[p, :k] = js_sorted[:k]
                    j_mask_comp[p, :k] = True


                comp_peaks.append(peaks)
                comp_integrals.append(nb_atoms * scales[i])
                comp_smiles.append(smiles)
                comp_j_vals.append(j_vals_comp)
                comp_j_mask.append(j_mask_comp)
            if not valid:
                continue  

            mix_peaks = np.concatenate(comp_peaks)
            mix_integrals = np.concatenate(comp_integrals)
            mix_j_vals = np.concatenate(comp_j_vals, axis=0)  
            mix_j_mask = np.concatenate(comp_j_mask, axis=0) 

            target = np.zeros(len(mix_peaks), dtype=np.int64)
            start = 0
            for comp_idx, peaks in enumerate(comp_peaks):
                n_peaks = len(peaks)
                target[start:start + n_peaks] = comp_idx 
                start += n_peaks

            num_peaks = len(mix_peaks)
            con = np.zeros((num_peaks, num_peaks), dtype=np.float32)

            for j in range(num_peaks):
                for k in range(num_peaks):
                    if mix_integrals[j] == 0 or mix_integrals[k] == 0:
                        print('I am in empty block')
                        con[j, k] = 0
                    else:
                        ratio = mix_integrals[j] / mix_integrals[k]
                        con[j, k] = ratio if ratio <= 1 else 1.0 / ratio

            np.fill_diagonal(con, 1.0)

            sort_idx = np.argsort(mix_peaks)
            mix_peaks = mix_peaks[sort_idx]
            mix_integrals = mix_integrals[sort_idx]
            mix_j_vals = mix_j_vals[sort_idx]
            mix_j_mask = mix_j_mask[sort_idx]
            target = target[sort_idx]
            con = con[sort_idx][:, sort_idx]

            return {
                "n_components": n_components, 
                "mix_peaks": torch.from_numpy(mix_peaks), # (P_mix)
                "mix_integrals": torch.from_numpy(mix_integrals), #(P_mix)
                "smiles": comp_smiles,
                "con": torch.from_numpy(con),  # (P_mix, P_mix)
                "target": torch.from_numpy(target),  #(n_components, P_individual_comp)
                "j_vals": torch.from_numpy(mix_j_vals),
                "j_mask": torch.from_numpy(mix_j_mask)      
                }

