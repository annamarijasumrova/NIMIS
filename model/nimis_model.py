import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment


class ScalarEmbedding(nn.Module):  
    def __init__(self, hidden_dim=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, pos, intensity):
        scalars = torch.stack([pos, intensity], dim=-1)  # (B,P,2)
        return self.net(scalars)                         # (B,P,out_dim)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class RatioAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout),
        )
        self.w_con = nn.Parameter(torch.tensor(1.0)) 
        self.w_conJ = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, con=None, mask=None, conj= None):
        """
        x:    (B, P, dim)
        con:  (B, P, P)  intensity ratio matrix
        mask: (B, P_max)
        """
        B, P, _ = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, h, P, P)
        if con is not None:
            con = con.unsqueeze(1).repeat(1, self.heads, 1, 1)
            dots = dots + self.w_con * con

        if conj is not None:
            conj = conj.unsqueeze(1).repeat(1, self.heads, 1, 1)
            dots = dots + self.w_conJ * conj

        # masking
        if mask is not None:
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,P)
            dots = dots.masked_fill(attn_mask, -1e9)

        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RatioAttention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, con=None, mask=None, conj=None):
        attn_maps = []
        for attn, ff in self.layers:
            o, attn_map = attn(x, con, mask, conj)
            x = x + o  # residual
            x = x + ff(x)
            attn_maps.append(attn_map)
        return x, attn_maps

class SlotAttention(nn.Module):

    def __init__(self, num_slots, dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters

        # mean+std of learned gaussian distribution// start random 
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        self.project_q = nn.Linear(dim, dim, bias=False)
        self.project_k = nn.Linear(dim, dim, bias=False)
        self.project_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, inputs, mask=None, component_mask=None):
        """
        inputs: (B, P, Dimension)
        mask:   (B, P_mix) 
        component_mask: (B, K_max)
        """

        B, P, D = inputs.shape
        inputs = self.norm_inputs(inputs)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = F.softplus(self.slots_sigma).expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        k = self.project_k(inputs)   # (B, P, D)
        v = self.project_v(inputs)   # (B, P, D)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.project_q(self.norm_slots(slots))   # (B, K, D)

            attn_logits = torch.einsum('bkd,bpd->bkp', q, k) / (D ** 0.5)
            if mask is not None:
                peak_mask = ~mask.unsqueeze(1)      # (B,1,P)
                attn_logits = attn_logits.masked_fill(peak_mask, -1e9)
            if component_mask is not None:
                attn_mask = ~component_mask.unsqueeze(-1)  # (B,K,1)
                attn_logits = attn_logits.masked_fill(attn_mask, -1e9)

            attn = F.softmax(attn_logits, dim=1)  # (B, K, P)
            updates = torch.einsum('bkp,bpd->bkd', attn, v)
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.num_slots, D)

            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn

class NMRPeakSlotAttn(nn.Module):
    def __init__(self, dim=128, mlp_dim=512, depth=6, heads=8, dim_head=64, K=4, slot_iters=3):

        super().__init__()
        self.K = K
        self.embedding = ScalarEmbedding(hidden_dim =64, out_dim=dim)
        self.dropout =nn.Dropout(0.1)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.slot_attention = SlotAttention(num_slots=K, dim=dim, iters=slot_iters)

    def forward(self, peaks, integrals, con, mask=None, component_mask=None, conj=None):
        """
        peaks: peaks (B,P_mix)
        integrals: integrals (B, P_mix)
        con: (B, P_mix, P_mix)
        mask: (B, P_mix) unmasks P
        component_mask:  (B, K_max) unmasks K
        """
        x = self.embedding(peaks, integrals)
        x = self.dropout(x)
        peak_features, attn_maps = self.transformer(x, con, mask, conj)
        slots, slot_attn = self.slot_attention(peak_features, mask, component_mask)
        return slots, slot_attn, peak_features, attn_maps


####################################
def build_con_J_hard_overlap(j_vals, j_mask, tol:float = 0.1):
    """
    j_vals: (B, P, 3)
    j_mask: (B, P, 3) bool
    """
    B, P, num_j = j_vals.shape
    assert num_j == 3

    J_p = j_vals.unsqueeze(2)   # (B, P, 1, 3)
    J_q = j_vals.unsqueeze(1)   # (B, 1, P, 3)

    M_p = j_mask.unsqueeze(2)   # (B, P, 1, 3)
    M_q = j_mask.unsqueeze(1)   # (B, 1, P, 3)

    valid = M_p & M_q           # (B, P, P, 3)

    eq = (torch.abs(J_p - J_q) <= tol) & valid   # (B, P, P, 3)

    con_J = eq.any(dim=-1).float()   # (B, P, P)

    has_J = j_mask.any(dim=-1).float()      # (B, P)
    con_J = con_J * has_J.unsqueeze(2) * has_J.unsqueeze(1)
    for b in range(B):
        con_J[b].fill_diagonal_(1.0 if has_J[b].any() else 0.0)

    return con_J

def collate128(batch, K_max=4, num_j=3, noise_std=0.10):
    B = len(batch)
    P_max = max(s["mix_peaks"].shape[0] for s in batch)   # max number of peaks per batch

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

        # con matrix (P_i x P_i): use precomputed if given, else derive from the same
        # (possibly noisy) integrals used above, so con and mix_integrals share one noise draw
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

    # Pad n_components to K_max
    component_mask = torch.zeros(B, K_max, dtype=torch.bool)
    for i, n in enumerate(n_components):
        component_mask[i, :n] = True

    return {
        "mask": masks,               # (B, P_max)
        "mix_peaks": mix_peaks,               # (B, P_max)
        "mix_integrals": mix_integrals,       # (B, P_max)
        "con": cons,                 # (B, P_max, P_max)
        "n_components": n_components,  # (B,)
        "component_mask": component_mask,  # (B, K_max)
        'targets': targets_padded,       #(B, P_max)
        'smiles': smiles,
        "j_vals": j_vals,       #(B, P_max, num_j(3))
        "j_mask": j_mask,        #(B, P_max, num_j(3))
        "con_J": conj,            #(B,P_max, P_max)
        "proportions": proportions,  # (B, K_max)
        "solvent": solvents,
        "frequency": frequencies,
    }


def slot_hungarian_loss(slot_attn, targets, mask, K):
    """
    slot_attn: (B, K, P)
    targets:   (B, P)
    mask:      (B, P)

    Same semantics as a per-sample Hungarian-matched cross-entropy, but built to avoid
    a GPU<->CPU sync per sample: the original nested Python loop called .item()/.cpu()
    up to B times per batch, which dominates wall-clock time far more than the model's
    actual compute (each sync blocks until the whole CUDA queue drains).
    """
    B, K_pred, P = slot_attn.shape
    assert K_pred == K
    device = slot_attn.device
    mask_bool = mask.bool()

    # one sync for the whole batch instead of one per sample
    targets_masked = targets.masked_fill(~mask_bool, -1)
    n_true_list = (targets_masked.amax(dim=1) + 1).clamp(min=1, max=K).cpu().tolist()
    total_peaks = int(mask_bool.sum().item())

    log_attn = torch.log(slot_attn + 1e-9)  # (B, K, P)

    BIG = 1e6
    cost_padded = torch.full((B, K, K), BIG, device=device)
    for b in range(B):
        n_true = n_true_list[b]
        valid = mask_bool[b]
        tgt_b = targets[b, valid]
        onehot = F.one_hot(tgt_b, num_classes=n_true).float()  # (P_valid, n_true)
        counts = onehot.sum(dim=0).clamp_min(1.0)               # (n_true,)
        cost = -(log_attn[b, :, valid] @ onehot) / counts       # (K, n_true), mean over peaks per (slot,true) pair
        cost_padded[b, :, :n_true] = cost

    # one device->host sync for the whole batch's cost matrices, not one per sample
    cost_np = cost_padded.detach().cpu().numpy()

    mapping_np = -np.ones((B, K), dtype=np.int64)      # slot -> true component
    inv_mapping_np = np.zeros((B, K), dtype=np.int64)  # true component -> slot
    for b in range(B):
        n_true = n_true_list[b]
        row_ind, col_ind = linear_sum_assignment(cost_np[b, :, :n_true])  # pure CPU/numpy, no GPU sync
        mapping_np[b, row_ind] = col_ind
        inv_mapping_np[b, col_ind] = row_ind

    mapping = torch.from_numpy(mapping_np).to(device)
    inv_mapping = torch.from_numpy(inv_mapping_np).to(device)

    assigned_slot = torch.gather(inv_mapping, 1, targets.clamp_min(0))              # (B, P)
    p_correct = torch.gather(slot_attn, 1, assigned_slot.unsqueeze(1)).squeeze(1)    # (B, P)
    p_correct = p_correct.clamp_min(1e-9)

    loss = (-torch.log(p_correct))[mask_bool].sum() / max(total_peaks, 1)
    return loss, mapping


def demo():
    torch.manual_seed(0)
    mask = torch.tensor([[True, True]])
    targets = torch.tensor([[0, 1]])

    # case 1: slots already aligned with true component order
    slot_attn = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    loss, mapping = slot_hungarian_loss(slot_attn, targets, mask, K=2)
    assert torch.allclose(loss, -torch.log(torch.tensor(0.9)), atol=1e-4)
    assert mapping.tolist() == [[0, 1]]

    # case 2: slots swapped relative to true component order -- Hungarian must undo it
    slot_attn = torch.tensor([[[0.1, 0.9], [0.9, 0.1]]])
    loss, mapping = slot_hungarian_loss(slot_attn, targets, mask, K=2)
    assert torch.allclose(loss, -torch.log(torch.tensor(0.9)), atol=1e-4)
    assert mapping.tolist() == [[1, 0]]

    print("slot_hungarian_loss self-check passed")


if __name__ == "__main__":
    demo()

def entropy_regularization(slot_attn, mask):
    eps = 1e-9
    ent = -(slot_attn * torch.log(slot_attn + eps)).sum(dim=1)  # (B, P)
    ent = ent[mask].mean()
    return ent
