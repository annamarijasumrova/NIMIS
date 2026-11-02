import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment


class ScalarEmbedding(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=128, n_freq=16, min_period=0.01, max_period=16.0):
        super().__init__()
        periods = torch.logspace(math.log10(max_period), math.log10(min_period), n_freq)
        self.register_buffer("freqs", 2 * math.pi / periods)
        self.net = nn.Sequential(
            nn.Linear(2 * n_freq + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pos, intensity):
        ang = pos.unsqueeze(-1) * self.freqs
        log_i = torch.log1p(intensity.clamp_min(0)).unsqueeze(-1)
        return self.net(torch.cat([ang.sin(), ang.cos(), log_i], dim=-1))


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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1, per_head_bias=False):
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
        shape = (heads,) if per_head_bias else ()
        self.w_con = nn.Parameter(torch.ones(shape))
        self.w_conJ = nn.Parameter(torch.ones(shape))

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

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if con is not None:
            w = self.w_con.view(1, -1, 1, 1) if self.w_con.dim() else self.w_con
            dots = dots + w * con.unsqueeze(1)

        if conj is not None:
            w = self.w_conJ.view(1, -1, 1, 1) if self.w_conJ.dim() else self.w_conJ
            dots = dots + w * conj.unsqueeze(1)

        if mask is not None:
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)
            dots = dots.masked_fill(attn_mask, -1e9)

        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, per_head_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RatioAttention(dim, heads=heads, dim_head=dim_head, per_head_bias=per_head_bias),
                FeedForward(dim, mlp_dim),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, con=None, mask=None, conj=None):
        attn_maps = []
        for attn, ff in self.layers:
            o, attn_map = attn(x, con, mask, conj)
            x = x + o
            x = x + ff(x)
            attn_maps.append(attn_map)
        return x, attn_maps

class SlotAttention(nn.Module):

    def __init__(self, num_slots, dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters

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

        k = self.project_k(inputs)
        v = self.project_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.project_q(self.norm_slots(slots))

            attn_logits = torch.einsum('bkd,bpd->bkp', q, k) / (D ** 0.5)
            if mask is not None:
                peak_mask = ~mask.unsqueeze(1)
                attn_logits = attn_logits.masked_fill(peak_mask, -1e9)
            if component_mask is not None:
                attn_mask = ~component_mask.unsqueeze(-1)
                attn_logits = attn_logits.masked_fill(attn_mask, -1e9)

            attn = F.softmax(attn_logits, dim=1)
            updates = torch.einsum('bkp,bpd->bkd', attn, v)
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.num_slots, D)

            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn

class NMRPeakSlotAttn(nn.Module):
    def __init__(self, dim=128, mlp_dim=512, depth=6, heads=8, dim_head=64, K=4, slot_iters=3, per_head_bias=False):

        super().__init__()
        self.K = K
        self.embedding = ScalarEmbedding(hidden_dim =64, out_dim=dim)
        self.dropout =nn.Dropout(0.1)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, per_head_bias)
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

    targets_masked = targets.masked_fill(~mask_bool, -1)
    n_true_list = (targets_masked.amax(dim=1) + 1).clamp(min=1, max=K).cpu().tolist()
    total_peaks = int(mask_bool.sum().item())

    log_attn = torch.log(slot_attn + 1e-9)

    BIG = 1e6
    cost_padded = torch.full((B, K, K), BIG, device=device)
    for b in range(B):
        n_true = n_true_list[b]
        valid = mask_bool[b]
        tgt_b = targets[b, valid]
        onehot = F.one_hot(tgt_b, num_classes=n_true).float()
        counts = onehot.sum(dim=0).clamp_min(1.0)
        cost = -(log_attn[b, :, valid] @ onehot) / counts
        cost_padded[b, :, :n_true] = cost

    cost_np = cost_padded.detach().cpu().numpy()

    mapping_np = -np.ones((B, K), dtype=np.int64)
    inv_mapping_np = np.zeros((B, K), dtype=np.int64)
    for b in range(B):
        n_true = n_true_list[b]
        row_ind, col_ind = linear_sum_assignment(cost_np[b, :, :n_true])
        mapping_np[b, row_ind] = col_ind
        inv_mapping_np[b, col_ind] = row_ind

    mapping = torch.from_numpy(mapping_np).to(device)
    inv_mapping = torch.from_numpy(inv_mapping_np).to(device)

    assigned_slot = torch.gather(inv_mapping, 1, targets.clamp_min(0))
    p_correct = torch.gather(slot_attn, 1, assigned_slot.unsqueeze(1)).squeeze(1)
    p_correct = p_correct.clamp_min(1e-9)

    loss = (-torch.log(p_correct))[mask_bool].sum() / max(total_peaks, 1)
    return loss, mapping


def demo():
    torch.manual_seed(0)
    mask = torch.tensor([[True, True]])
    targets = torch.tensor([[0, 1]])

    slot_attn = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    loss, mapping = slot_hungarian_loss(slot_attn, targets, mask, K=2)
    assert torch.allclose(loss, -torch.log(torch.tensor(0.9)), atol=1e-4)
    assert mapping.tolist() == [[0, 1]]

    slot_attn = torch.tensor([[[0.1, 0.9], [0.9, 0.1]]])
    loss, mapping = slot_hungarian_loss(slot_attn, targets, mask, K=2)
    assert torch.allclose(loss, -torch.log(torch.tensor(0.9)), atol=1e-4)
    assert mapping.tolist() == [[1, 0]]

    print("slot_hungarian_loss self-check passed")


if __name__ == "__main__":
    demo()

def entropy_regularization(slot_attn, mask):
    eps = 1e-9
    ent = -(slot_attn * torch.log(slot_attn + eps)).sum(dim=1)
    ent = ent[mask].mean()
    return ent
