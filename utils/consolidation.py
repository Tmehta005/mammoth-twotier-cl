"""
Consolidation strats for two tier DER++

Each method scores FIFO samples and returns index ordering best first
used to prioritze which samples get pusheed into LTM
"""

import torch
import torch.nn.functional as F

def _extract_features(net, x):
    """Extract features from Mammoth"""
    try:
        return net(x, returnt='features')
    except TypeError:
        return net(x)
    
def get_consolidation_fn(strategy: str):
    """Return the consolidation function for the given implementation we use"""
    registry = {
        'random': consolidate_random,
        'diversity': consolidate_diversity,
        'loss': consolidate_loss,
        'hybrid': consolidate_hybrid,
    }
    if strategy not in registry:
        raise ValueError(
            f"Unknown consolidation strategy '{strategy}'. "
            f"Choose from: {list(registry.keys())}")
    return registry[strategy]


def consolidate_random(stm, ltm, net, device):
    """No reordering, so push all FIFO samples in original order"""
    n = stm._filled_size()
    return torch.arange(n, device=device)


@torch.no_grad()
def consolidate_diversity(stm, ltm, net, device, n_ref_samples=100):
    """Diversity-based reordering for FIFO samples"""
    stm_ex, _, _ = stm.get_filled_data()
    if stm_ex is None:
        return torch.arange(0, device=device)

    n = stm_ex.size(0)
    was_training = net.training
    net.eval()

    stm_feats = _extract_features(net, stm_ex.to(device))
    stm_feats = F.normalize(stm_feats, dim=1)

    if not ltm.is_empty():
        n_ref = min(n_ref_samples, ltm.num_seen_examples)
        ref_ex, _, _ = ltm.get_data(n_ref, device=device)
        ref_feats = _extract_features(net, ref_ex)
        ref_feats = F.normalize(ref_feats, dim=1)
    else:
        ref_feats = None

    selected = []
    remaining = set(range(n))

    for _ in range(n):
        if not remaining:
            break

        remaining_idx = torch.tensor(list(remaining), device=device)

        if ref_feats is not None and ref_feats.size(0) > 0:
            sim = stm_feats[remaining_idx] @ ref_feats.T
            max_sim, _ = sim.max(dim=1)
            pick_pos = max_sim.argmin().item()
        else:
            pick_pos = 0

        pick_idx = remaining_idx[pick_pos].item()
        selected.append(pick_idx)
        remaining.discard(pick_idx)

        new_feat = stm_feats[pick_idx:pick_idx + 1]
        if ref_feats is not None:
            ref_feats = torch.cat([ref_feats, new_feat], dim=0)
        else:
            ref_feats = new_feat

    if was_training:
        net.train()

    return torch.tensor(selected, device=device)


@torch.no_grad()
def consolidate_loss(stm, ltm, net, device):
    """Reorder FIFO samples based on highest CE loss"""
    stm_ex, stm_lb, _ = stm.get_filled_data()
    if stm_ex is None or stm_lb is None:
        return torch.arange(0, device=device)

    was_training = net.training
    net.eval()

    logits = net(stm_ex.to(device))
    losses = F.cross_entropy(logits, stm_lb.to(device), reduction='none')
    order = losses.argsort(descending=True)

    if was_training:
        net.train()

    return order


@torch.no_grad()
def consolidate_hybrid(stm, ltm, net, device, n_ref_samples=100):
    """Combined ranking: score = loss_rank + diversity_rank."""
    stm_ex, stm_lb, _ = stm.get_filled_data()
    if stm_ex is None or stm_lb is None:
        return torch.arange(0, device=device)

    n = stm_ex.size(0)
    was_training = net.training
    net.eval()

    stm_ex_d = stm_ex.to(device)
    stm_lb_d = stm_lb.to(device)

    # Loss ranking where rank 0 is highest loss
    logits = net(stm_ex_d)
    losses = F.cross_entropy(logits, stm_lb_d, reduction='none')
    loss_ranks = losses.argsort(descending=True).argsort().float()

    # Diversity ranking where rank 0 is most distant from LTM
    feats = _extract_features(net, stm_ex_d)
    feats = F.normalize(feats, dim=1)

    if not ltm.is_empty():
        n_ref = min(n_ref_samples, ltm.num_seen_examples)
        ref_ex, _, _ = ltm.get_data(n_ref, device=device)
        ref_feats = _extract_features(net, ref_ex)
        ref_feats = F.normalize(ref_feats, dim=1)
        sim = feats @ ref_feats.T
        max_sim, _ = sim.max(dim=1)
        div_ranks = max_sim.argsort().argsort().float()
    else:
        div_ranks = torch.zeros(n, device=device)

    # Combined score 
    combined = loss_ranks + div_ranks
    order = combined.argsort()

    if was_training:
        net.train()

    return order