import torch


def build_con_J_hard_overlap(j_vals, j_mask, tol:float = 0.1):
    """
    j_vals: (B, P, 3)
    j_mask: (B, P, 3) bool
    """
    B, P, num_j = j_vals.shape
    assert num_j == 3

    J_p = j_vals.unsqueeze(2)
    J_q = j_vals.unsqueeze(1)

    M_p = j_mask.unsqueeze(2)
    M_q = j_mask.unsqueeze(1)

    valid = M_p & M_q

    eq = (torch.abs(J_p - J_q) <= tol) & valid

    con_J = eq.any(dim=-1).float()

    has_J = j_mask.any(dim=-1).float()
    con_J = con_J * has_J.unsqueeze(2) * has_J.unsqueeze(1)
    con_J.diagonal(dim1=1, dim2=2).fill_(1.0)

    return con_J

def collate128(batch, K_max=4, num_j=3, noise_std=0.10):
    B = len(batch)
    P_max = max(s["mix_peaks"].shape[0] for s in batch)

    masks = torch.zeros(B, P_max, dtype=torch.bool)
    cons = torch.zeros(B, P_max, P_max, dtype=torch.float32)
    n_components = torch.zeros(B, dtype=torch.long)
    targets_padded = torch.full((B, P_max), -100, dtype=torch.long)
    mix_peaks = torch.zeros(B, P_max, dtype=torch.float32)
    mix_integrals = torch.zeros(B, P_max, dtype=torch.float32)
    j_vals = torch.zeros(B, P_max, num_j, dtype=torch.float32)
    j_mask = torch.zeros(B, P_max, num_j, dtype=torch.bool)
    proportions = torch.zeros(B, K_max, dtype=torch.float32)
    smiles = []
    solvents = []
    frequencies = []

    for i, sample in enumerate(batch):
        mixture = sample["mix_peaks"]
        target = sample["target"]
        P_i = mixture.shape[0]
        mix_peaks[i, :P_i] = mixture
        integ = torch.as_tensor(sample["mix_integrals"], dtype=torch.float32)
        if noise_std > 0:
            integ = integ * (1.0 + torch.randn(P_i) * noise_std)
        mix_integrals[i, :P_i] = integ
        masks[i, :P_i] = True

        if "j_vals" in sample and "j_mask" in sample:
            J = torch.as_tensor(sample["j_vals"], dtype=torch.float32)
            M = torch.as_tensor(sample["j_mask"], dtype=torch.bool)
            j_vals[i, :P_i, :] = J
            j_mask[i, :P_i, :] = M


        s = sample["smiles"]
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8")
        elif isinstance(s, (list, tuple)):
            s = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x for x in s]
        smiles.append(s)

        if sample.get("con") is not None:
            cons[i, :P_i, :P_i] = sample["con"]
        else:
            a = integ.unsqueeze(0)
            b = integ.unsqueeze(1)
            ratio = torch.minimum(a, b) / torch.maximum(a, b).clamp_min(1e-9)
            ratio = torch.where((a == 0) | (b == 0), torch.zeros_like(ratio), ratio)
            ratio.fill_diagonal_(1.0)
            cons[i, :P_i, :P_i] = ratio

        targets_padded[i, :P_i] = target
        n_components[i] = sample["n_components"]

        scales = sample.get("scales")
        if scales is not None:
            n = min(len(scales), K_max)
            proportions[i, :n] = torch.as_tensor(scales[:n], dtype=torch.float32)

        solvents.append(sample.get("solvent"))
        frequencies.append(sample.get("frequency"))

    conj = build_con_J_hard_overlap(j_vals, j_mask, tol=0.1)

    component_mask = torch.zeros(B, K_max, dtype=torch.bool)
    for i, n in enumerate(n_components):
        component_mask[i, :n] = True

    return {
        "mask": masks,
        "mix_peaks": mix_peaks,
        "mix_integrals": mix_integrals,
        "con": cons,
        "n_components": n_components,
        "component_mask": component_mask,
        'targets': targets_padded,
        'smiles': smiles,
        "j_vals": j_vals,
        "j_mask": j_mask,
        "con_J": conj,
        "proportions": proportions,
        "solvent": solvents,
        "frequency": frequencies,
    }
