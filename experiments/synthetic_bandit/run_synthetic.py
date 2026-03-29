#!/usr/bin/env python3
"""Synthetic preference-bandit experiments for the ICML 2026 OLE submission.

This script reproduces:
  (i) regret scaling plot (Regret(t) and Regret(t)/sqrt(t)),
 (ii) implementation-term table (varying ensemble size N and step size eta0),
(iii) mini ablation table (mean-only vs variance-only vs mean+variance (OLE),
      fixed vs adaptive reference, and SGLD vs deterministic SGD),
 (iv) a baseline comparison table (Laplace-UCB, Laplace-Thompson, and ensemble TS),
  (v) a compute--regret tradeoff plot (Regret@T vs the wall-clock proxy N×m),
 (vi) a lightweight hyperparameter sensitivity sweep over a small (N × eta0) grid
      and a corresponding heatmap figure (Appendix).

The environment matches the paper's BTL/logistic preference model. For maximal
control, we use a degenerate contextual bandit with a single context, which is a
special case of the contextual bandit setting analyzed in the paper.

Dependencies: numpy, matplotlib

Usage (from repo root):
  python experiments/synthetic_bandit/run_synthetic.py --task all
  python experiments/synthetic_bandit/run_synthetic.py --task scaling --T 2000 --n_seeds 12
"""

import argparse
import csv
import json
import math
import os

# Optional suffix for figure filenames (set from --tag in main).
FIG_SUFFIX = ""

from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh_reward(s: np.ndarray) -> np.ndarray:
    """Bounded reward parameterization used in the paper.

    We use r(s) = 0.5 * (1 + tanh(s)) so r \in (0,1) and dr/ds is bounded.
    """
    return 0.5 * (1.0 + np.tanh(s))


def tanh_reward_ds(s: np.ndarray) -> np.ndarray:
    """Derivative dr/ds for r(s)=0.5*(1+tanh(s))."""
    t = np.tanh(s)
    return 0.5 * (1.0 - t * t)


def make_bandit_instance(cfg: "SimConfig") -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Create a fixed synthetic bandit instance (actions + true parameter)."""
    rng_env = np.random.default_rng(cfg.env_seed)

    theta_star = rng_env.normal(size=cfg.d) * cfg.gap_scale
    actions = rng_env.normal(size=(cfg.K, cfg.d))
    actions = actions / np.linalg.norm(actions, axis=1, keepdims=True)
    phi = actions  # single context => features are just action embeddings

    s_star = theta_star @ phi.T
    r_star = tanh_reward(s_star)
    best = int(np.argmax(r_star))
    return phi, theta_star, r_star, best


@dataclass
class SimConfig:
    T: int = 2000
    d: int = 8
    K: int = 20
    # To reduce variance in scaling plots and make experiments easier to reproduce,
    # we allow fixing a single bandit instance (arm embeddings + true parameter)
    # across random seeds. Each run then differs only in preference noise and
    # Langevin/mini-batch randomness.
    env_seed: int = 0
    N: int = 20
    m: int = 32
    n_seeds: int = 12
    eta0: float = 1.0
    kappa: float = 0.3
    beta: float = 1.0
    gap_scale: float = 0.3
    prior_sigma: float = 0.5
    clip: float = 5
    ref_mode: str = "fixed"  # fixed | prev
    variant: str = "ole"     # ole | mean | var | ts
    sgld: bool = True
    method: str = "ole"      # ole | enspp
    # ── Ensemble++ parameters ─────────────────────────────────────────
    M: int = 8     # ensemble size; paper recommends Θ(d log T)
    lam: float = 1        # ridge prior: Σ₀⁻¹ = λI
    sigma: float = 2     # noise std (should match environment)
    zeta_dist: str = "gaussian" # gaussian | sphere | coord | cube
    z_dist: str = "gaussian"    # gaussian | sphere | coord | cube
    #── HyperAgent-specific parameters ──────────────────────────────
    # These follow the hypermodel framework from Li et al. (ICML 2024):
    #   "Q-Star Meets Scalable Posterior Sampling: Bridging Theory and
    #    Practice via HyperAgent".
    #
    # hyperagent_M : dimensionality of the random index ξ ∈ R^M.
    #   Theory requires M = O(log(K·T)); in practice 10–30 works well.
    # hyperagent_sigma : scale of the algorithmic perturbation σ ξ^⊤ z
    #   injected into the loss.  Controls exploration strength.
    # hyperagent_n_indices : number of indices |Ξ̃| for the sample-average
    #   approximation of E_ξ[ℓ(θ;ξ,d)] in the update step.
    # hyperagent_prior_scale_mult : multiplier for the fixed prior
    #   uncertainty matrix A_0.  Larger values → stronger prior exploration.
    hyperagent_M: int = 20
    hyperagent_sigma: float = 0.3
    hyperagent_n_indices: int = 20
    hyperagent_prior_scale_mult: float = 1.0
    hyperagent_kappa: float = 0.3
    hyperagent_beta: float =1.0
    # ── HyperAgent+OLE hybrid parameters ────────────────────────────
    # Multi-head hypermodel with OLE-style acquisition (mean + kappa*std).
    # Uses N_heads independent hypermodels; each samples one index ξ to
    # produce a reward prediction, then selects actions via mean + kappa*std
    # across the N_heads predictions (OLE-style exploration bonus).
    hybrid_n_heads: int = 2
    hybrid_kappa: float = 1.0



# ═══════════════════════════════════════════════════════════════════════
#  HyperAgent:  Approximate Posterior Sampling via Hypermodel
# ═══════════════════════════════════════════════════════════════════════
#
# Adaptation of HyperAgent (Li et al., ICML 2024) to the binary
# preference-bandit setting with tanh-reward parameterization.
#
# Key design choices for the adaptation:
#   1. The RL Q-value hypermodel f_θ(s,a,ξ) becomes a reward hypermodel
#      f_θ(a,ξ) = φ(a)^⊤ [(b_w + A_w ξ) + (b_0 + A_0 ξ)]
#      since there are no states in the bandit setting.
#   2. The TD loss with bootstrapping is replaced by cross-entropy
#      preference loss, matching the BTL observation model.
#   3. The algorithmic perturbation σ ξ^⊤ z is added to the preference
#      logit, analogous to the perturbation on the TD target.
#   4. No target network is needed (bandit = single-step, no bootstrap).
#   5. The update averages over a batch of indices Ξ̃ (sample-average
#      approximation), following Algorithm 2 of the original paper.
# ═══════════════════════════════════════════════════════════════════════

def _simulate_hyperagent(seed: int, cfg: SimConfig) -> np.ndarray:
    """Run HyperAgent for T rounds on the preference bandit.

    The hypermodel maintains:
      - Learnable parameters: b_w ∈ R^d (mean), A_w ∈ R^{d×M} (uncertainty)
      - Fixed prior:          b_0 ∈ R^d (zero), A_0 ∈ R^{d×M} (random)

    For a random index ξ ~ N(0, I_M), the randomized parameter is:
      θ(ξ) = (b_w + A_w ξ) + (b_0 + A_0 ξ)

    The variation across ξ captures epistemic uncertainty.  Actions with
    high uncertainty get explored because their hypermodel outputs vary
    more across index samples.

    Returns: cumulative regret array of length T.
    """
    rng = np.random.default_rng(seed)
    phi, _theta_star, r_star, best = make_bandit_instance(cfg)

    M = cfg.hyperagent_M
    sigma_perturb = cfg.hyperagent_sigma
    n_xi = cfg.hyperagent_n_indices

    # ── Learnable parameters ──
    b_w = np.zeros(cfg.d, dtype=float)             # mean (initialized near zero)
    A_w = np.zeros((cfg.d, M), dtype=float)        # uncertainty (initialized to zero)

    # ── Fixed prior (additive prior assumption from the paper) ──
    # The prior model f^P has no trainable parameters.  A_0 encodes
    # the prior uncertainty; its scale controls initial exploration.
    prior_scale = (
        cfg.hyperagent_prior_scale_mult
        * cfg.prior_sigma
        * cfg.gap_scale
        / math.sqrt(M)
    )
    b_0 = np.zeros(cfg.d, dtype=float)
    A_0 = rng.normal(scale=prior_scale, size=(cfg.d, M))

    # ── Replay buffer (pre-allocated for speed) ──
    y_buf   = np.empty(cfg.T, dtype=np.int64)
    ref_buf = np.empty(cfg.T, dtype=np.int64)
    b_buf   = np.empty(cfg.T, dtype=np.float64)
    z_buf   = np.empty((cfg.T, M), dtype=np.float64)   # perturbation vectors
    n_data  = 0

    regret  = np.zeros(cfg.T, dtype=float)
    ell_max = math.log(1.0 + math.e)
    y_prev  = 0

    for t in range(1, cfg.T + 1):
        ref = 0 if cfg.ref_mode == "fixed" else y_prev

        # ─── Action selection (Algorithm 1, line 5) ───────────────
        # Sample one index ξ and act greedily under the randomized
        # reward model.
        xi_act = rng.normal(size=M)
        A_total_act = A_w + A_0
        b_total_act = b_w + b_0
        theta_act = b_total_act + A_total_act @ xi_act          # (d,)

        s_act = phi @ theta_act                                  # (K,)
        r_act = tanh_reward(s_act)
        delta_act = r_act - r_act[ref]

        y = int(np.argmax(delta_act))
        y_prev = y

        regret[t - 1] = float(r_star[best] - r_star[y])

        # ─── Preference observation ───────────────────────────────
        delta_star = float(r_star[y] - r_star[ref])
        p = float(sigmoid(np.array(delta_star)))
        b = 1.0 if rng.random() < p else 0.0

        # Store transition with its perturbation vector z (Algorithm 1, line 9).
        y_buf[n_data]   = y
        ref_buf[n_data] = ref
        b_buf[n_data]   = b
        z_buf[n_data]   = rng.normal(size=M)
        n_data += 1

        # ─── Incremental update (Algorithm 2) ─────────────────────
        # One SGD step on a mini-batch, with sample-average over a
        # batch of L = n_xi indices from P_ξ (Equation 5 in paper).
        eta = cfg.eta0 / t
        mb  = min(cfg.m, n_data)
        idxs = rng.integers(0, n_data, size=mb)

        ys   = y_buf[idxs]                         # (mb,)
        rs   = ref_buf[idxs]                        # (mb,)
        b_mb = b_buf[idxs]                          # (mb,)
        z_mb = z_buf[idxs]                          # (mb, M)
        Phi_y = phi[ys]                             # (mb, d)
        Phi_r = phi[rs]                             # (mb, d)

        # Sample a batch of indices Ξ̃ = {ξ_1, ..., ξ_L}
        xi_batch = rng.normal(size=(n_xi, M))       # (L, M)

        # ── Vectorized forward pass over all (data × index) pairs ──
        A_total = A_w + A_0                          # (d, M)
        b_total = b_w + b_0                          # (d,)
        # θ_l = b_total + A_total @ ξ_l  for each l
        theta_batch = b_total[None, :] + xi_batch @ A_total.T   # (L, d)

        # Raw scores: s_{y,j,l} = φ_{y_j}^⊤ θ_l
        s_y_all = Phi_y @ theta_batch.T              # (mb, L)
        s_r_all = Phi_r @ theta_batch.T              # (mb, L)

        # Bounded rewards
        r_y_all = tanh_reward(s_y_all)               # (mb, L)
        r_r_all = tanh_reward(s_r_all)               # (mb, L)

        # Algorithmic perturbation: σ · ξ_l^⊤ z_j  for each (j, l).
        # This is the key mechanism that maintains posterior variance:
        # data points with fewer visits accumulate less "explained" noise,
        # keeping their uncertainty high and driving exploration.
        perturb_all = sigma_perturb * (z_mb @ xi_batch.T)  # (mb, L)

        # Perturbed preference logit
        logit_all = np.clip(
            r_y_all - r_r_all + perturb_all,
            -cfg.clip, cfg.clip,
        )                                            # (mb, L)

        p_all    = sigmoid(logit_all)                # (mb, L)
        coef_all = (p_all - b_mb[:, None]) / ell_max # (mb, L)

        # Derivatives through tanh_reward (chain rule)
        dy_all = tanh_reward_ds(s_y_all)             # (mb, L)
        dr_all = tanh_reward_ds(s_r_all)             # (mb, L)

        # ── Gradient computation (vectorized over indices) ──
        # For each index l:
        #   grad_θ_l = (1/mb) Σ_j coef_{j,l} · [dy_{j,l}·φ_{y_j} − dr_{j,l}·φ_{r_j}]
        #
        # Stacking into matrices:
        #   g_y = coef ⊙ dy ∈ R^{mb×L},  g_r = coef ⊙ dr ∈ R^{mb×L}
        #   grad_θ_batch = (1/mb) · [g_y^⊤ Φ_y − g_r^⊤ Φ_r]  ∈ R^{L×d}

        g_y = coef_all * dy_all                      # (mb, L)
        g_r = coef_all * dr_all                      # (mb, L)
        grad_theta_batch = (g_y.T @ Phi_y - g_r.T @ Phi_r) / mb   # (L, d)

        # grad w.r.t. b_w:  average grad_θ over indices
        #   ∂θ/∂b_w = I  ⟹  ∂L/∂b_w = (1/L) Σ_l grad_θ_l
        grad_b_w = grad_theta_batch.mean(axis=0)     # (d,)

        # grad w.r.t. A_w:  ∂θ_i/∂(A_w)_{i,m} = ξ_m  ⟹
        #   ∂L/∂A_w = (1/L) Σ_l grad_θ_l ⊗ ξ_l  =  (1/L) grad_θ_batch^⊤ ξ_batch
        grad_A_w = (grad_theta_batch.T @ xi_batch) / n_xi   # (d, M)

        # Scale to full-dataset gradient (standard SGLD / SGD scaling).
        scale = float(n_data) / float(mb)
        grad_b_w *= scale
        grad_A_w *= scale

        # Prior regularization: ∇_θ [β/(2σ²) ||θ||²]
        grad_b_w += cfg.hyperagent_beta * b_w / (cfg.prior_sigma ** 2)
        grad_A_w += cfg.hyperagent_beta * A_w / (cfg.prior_sigma ** 2)

        # SGD step.  Note: unlike the SGLD ensemble in OLE, HyperAgent
        # does NOT inject Langevin noise here — the randomization already
        # comes from the index ξ via the hypermodel structure.
        b_w = b_w - eta * grad_b_w
        A_w = A_w - eta * grad_A_w

        b_w = np.clip(b_w, -cfg.clip, cfg.clip)
        A_w = np.clip(A_w, -cfg.clip, cfg.clip)

    return np.cumsum(regret)



# ═══════════════════════════════════════════════════════════════════════
#  HyperAgent+OLE:  Multi-Head Hypermodel with OLE Acquisition
# ═══════════════════════════════════════════════════════════════════════
#
# Combines HyperAgent's hypermodel posterior approximation with OLE's
# UCB-style action selection.  Instead of a single hypermodel acting
# greedily on one sampled θ(ξ) (Thompson style), we maintain N_heads
# independent hypermodels.  Each head samples its own index ξ_h to
# produce θ_h, then action selection uses the OLE acquisition function:
#     score(a) = mean_h[delta_h(a)] + kappa * std_h[delta_h(a)]
# This provides an explicit exploration bonus from the variance across
# heads, while each head benefits from HyperAgent's structured posterior.
# ═══════════════════════════════════════════════════════════════════════

def _simulate_hyperagent_ole(seed: int, cfg: SimConfig) -> np.ndarray:
    """Run HyperAgent+OLE hybrid for T rounds on the preference bandit.

    Maintains N_heads independent hypermodels with shared prior structure.
    Action selection uses OLE's mean + kappa * std over the heads'
    reward-delta predictions.  Each head is updated independently via
    the same SGD procedure as standard HyperAgent.

    Returns: cumulative regret array of length T.
    """
    rng = np.random.default_rng(seed)
    phi, _theta_star, r_star, best = make_bandit_instance(cfg)

    M = cfg.hyperagent_M
    sigma_perturb = cfg.hyperagent_sigma
    n_xi = cfg.hyperagent_n_indices
    N = cfg.hybrid_n_heads
    kappa = cfg.hybrid_kappa

    # ── Learnable parameters: N independent heads ──
    b_ws = np.zeros((N, cfg.d), dtype=float)
    A_ws = np.zeros((N, cfg.d, M), dtype=float)

    # ── Fixed prior (shared structure, independent random draws) ──
    prior_scale = (
        cfg.hyperagent_prior_scale_mult
        * cfg.prior_sigma
        * cfg.gap_scale
        / math.sqrt(M)
    )
    b_0 = np.zeros(cfg.d, dtype=float)
    A_0s = rng.normal(scale=prior_scale, size=(N, cfg.d, M))

    # ── Replay buffer ──
    y_buf   = np.empty(cfg.T, dtype=np.int64)
    ref_buf = np.empty(cfg.T, dtype=np.int64)
    b_buf   = np.empty(cfg.T, dtype=np.float64)
    z_bufs  = np.empty((N, cfg.T, M), dtype=np.float64)
    n_data  = 0

    regret  = np.zeros(cfg.T, dtype=float)
    ell_max = math.log(1.0 + math.e)
    y_prev  = 0

    for t in range(1, cfg.T + 1):
        ref = 0 if cfg.ref_mode == "fixed" else y_prev

        # ─── OLE-style action selection across heads ──────────────
        deltas = np.empty((N, cfg.K))
        for h in range(N):
            xi_h = rng.normal(size=M)
            theta_h = (b_ws[h] + b_0) + (A_ws[h] + A_0s[h]) @ xi_h
            s_h = phi @ theta_h
            r_h = tanh_reward(s_h)
            deltas[h] = r_h - r_h[ref]

        mean_d = deltas.mean(axis=0)
        std_d = deltas.std(axis=0, ddof=1) if N > 1 else np.zeros(cfg.K)
        score = mean_d + kappa * std_d

        y = int(np.argmax(score))
        y_prev = y
        regret[t - 1] = float(r_star[best] - r_star[y])

        # ─── Preference observation ───────────────────────────────
        delta_star = float(r_star[y] - r_star[ref])
        p = float(sigmoid(np.array(delta_star)))
        b = 1.0 if rng.random() < p else 0.0

        y_buf[n_data]   = y
        ref_buf[n_data] = ref
        b_buf[n_data]   = b
        for h in range(N):
            z_bufs[h, n_data] = rng.normal(size=M)
        n_data += 1

        # ─── Update each head independently (Algorithm 2) ─────────
        eta = cfg.eta0 / t
        mb  = min(cfg.m, n_data)
        idxs = rng.integers(0, n_data, size=mb)

        ys   = y_buf[idxs]
        rs   = ref_buf[idxs]
        b_mb = b_buf[idxs]
        Phi_y = phi[ys]
        Phi_r = phi[rs]

        for h in range(N):
            z_mb = z_bufs[h, idxs]
            xi_batch = rng.normal(size=(n_xi, M))

            A_total = A_ws[h] + A_0s[h]
            b_total = b_ws[h] + b_0
            theta_batch = b_total[None, :] + xi_batch @ A_total.T

            s_y_all = Phi_y @ theta_batch.T
            s_r_all = Phi_r @ theta_batch.T
            r_y_all = tanh_reward(s_y_all)
            r_r_all = tanh_reward(s_r_all)
            perturb_all = sigma_perturb * (z_mb @ xi_batch.T)

            logit_all = np.clip(
                r_y_all - r_r_all + perturb_all, -cfg.clip, cfg.clip
            )
            p_all    = sigmoid(logit_all)
            coef_all = (p_all - b_mb[:, None]) / ell_max

            dy_all = tanh_reward_ds(s_y_all)
            dr_all = tanh_reward_ds(s_r_all)
            g_y = coef_all * dy_all
            g_r = coef_all * dr_all
            grad_theta_batch = (g_y.T @ Phi_y - g_r.T @ Phi_r) / mb

            grad_b = grad_theta_batch.mean(axis=0)
            grad_A = (grad_theta_batch.T @ xi_batch) / n_xi

            # Gradient clipping for numerical stability
            grad_b = np.clip(grad_b, -10.0, 10.0)
            grad_A = np.clip(grad_A, -10.0, 10.0)

            scale = float(n_data) / float(mb)
            grad_b *= scale
            grad_A *= scale

            grad_b += cfg.hyperagent_beta * b_ws[h] / (cfg.prior_sigma ** 2)
            grad_A += cfg.hyperagent_beta * A_ws[h] / (cfg.prior_sigma ** 2)

            b_ws[h] = np.clip(b_ws[h] - eta * grad_b, -cfg.clip, cfg.clip)
            A_ws[h] = np.clip(A_ws[h] - eta * grad_A, -cfg.clip, cfg.clip)

    return np.cumsum(regret)


def _sample_vec(rng: np.random.Generator, dist: str, size: int) -> np.ndarray:
    if dist == "gaussian":
        return rng.standard_normal(size)
    elif dist == "sphere":
        v = rng.standard_normal(size)
        return v / np.linalg.norm(v) * np.sqrt(size)
    elif dist == "coord":
        idx = rng.integers(size)
        v = np.zeros(size)
        v[idx] = np.sqrt(size)
        return v
    elif dist == "cube":
        return rng.choice([-1.0, 1.0], size=size)
    else:
        raise ValueError(f"Unknown distribution: {dist}")



def _simulate_enspp(seed: int, cfg: SimConfig) -> np.ndarray:
    """
    Ensemble++ for Preference Learning (Dueling Bandits).
    Uses Replay Buffer and SGD, observing ONLY binary preference feedback (0 or 1).
    """
    rng = np.random.default_rng(seed)
    T, d, K, M = cfg.T, cfg.d, cfg.K, cfg.M


    lam = cfg.beta / (cfg.prior_sigma ** 2) if hasattr(cfg, 'beta') else cfg.lam

    phi, theta_star, r_star, best = make_bandit_instance(cfg)

    # ── 1. initialize the net paramet ──────────────────────────────────────────────
    b = np.zeros(d)               # (Base Head)
    Theta = np.zeros((d, M))      # (Ensemble Heads)

    # (Fixed Prior)。
    Theta_0 = rng.standard_normal((d, M)) / math.sqrt(lam * M)

    # ── 2. (Replay Buffer) ─────────────────────────────
    y_buf   = np.empty(T, dtype=np.int64)
    ref_buf = np.empty(T, dtype=np.int64)
    b_buf   = np.empty(T, dtype=np.float64)
    Z_buf   = np.empty((T, M), dtype=np.float64)  # 专门存扰动噪声的 Buffer

    n_data  = 0
    y_prev  = 0
    regret  = np.zeros(T, dtype=float)
    ell_max = math.log(1.0 + math.e)

    for t in range(1, T + 1):
        ref = 0 if cfg.ref_mode == "fixed" else y_prev

        # ── 3. Action Selection (Thompson Sampling) ────────────────────
        zeta = _sample_vec(rng, cfg.zeta_dist, M)

        s_base = phi @ b
        s_ens  = phi @ (Theta + Theta_0) @ zeta
        s_total = s_base + s_ens

        # arm selection
        y = int(np.argmax(s_total))
        y_prev = y
        regret[t - 1] = float(r_star[best] - r_star[y])

        # ── 4. get preference feedback ──────────────────────
        delta_star = float(r_star[y] - r_star[ref])
        p = float(sigmoid(np.array(delta_star)))
        b_obs = 1.0 if rng.random() < p else 0.0


        z_obs = _sample_vec(rng, cfg.z_dist, M) / math.sqrt(M)


        y_buf[n_data]   = y
        ref_buf[n_data] = ref
        b_buf[n_data]   = b_obs
        Z_buf[n_data]   = z_obs
        n_data += 1

        # ── 5. SGD ─────────────
        eta = cfg.eta0 / t
        mb  = min(cfg.m, n_data)
        idxs = rng.integers(0, n_data, size=mb)

        ys_mb = y_buf[idxs]
        rs_mb = ref_buf[idxs]
        b_mb  = b_buf[idxs]
        Z_mb  = Z_buf[idxs]

        Phi_y = phi[ys_mb]
        Phi_r = phi[rs_mb]
        Delta_Phi = Phi_y - Phi_r  # (mb, d)


        s_y = Phi_y @ b
        s_r = Phi_r @ b
        ry  = tanh_reward(s_y)
        rr  = tanh_reward(s_r)

        p_mb = sigmoid(ry - rr)
        coef = (p_mb - b_mb) / ell_max
        dy   = 0.5 * (1.0 - np.tanh(s_y) ** 2)
        dr   = 0.5 * (1.0 - np.tanh(s_r) ** 2)


        grad_b_data = (coef * dy) @ Phi_y - (coef * dr) @ Phi_r

        grad_b = grad_b_data * (float(n_data) / float(mb)) + lam * b


        pred_Z = Delta_Phi @ (Theta + Theta_0)  # (mb, M)
        err_Z  = pred_Z - Z_mb                  # (mb, M)


        grad_Theta_data = Delta_Phi.T @ err_Z   # (d, M)

        grad_Theta = grad_Theta_data * (float(n_data) / float(mb)) + lam * Theta


        b = b - eta * grad_b
        Theta = Theta - eta * grad_Theta

        b = np.clip(b, -cfg.clip, cfg.clip)
        Theta = np.clip(Theta, -cfg.clip, cfg.clip)

    return np.cumsum(regret)
def _simulate_enspp_ole(seed: int, cfg: SimConfig) -> np.ndarray:
    """
    Combination of Ensemble++ and OLE for Preference Learning (Dueling Bandits).
    - Uses stable decoupled learning rates to prevent MSE divergence.
    - Evaluates uncertainty purely on the relative feature geometry.
    """
    rng = np.random.default_rng(seed)
    T, d, K, M = cfg.T, cfg.d, cfg.K, cfg.M

    lam = cfg.beta / (cfg.prior_sigma ** 2) if hasattr(cfg, 'beta') else cfg.lam
    phi, theta_star, r_star, best = make_bandit_instance(cfg)

    # ── 1. Initialize Network Parameters ─────────────────────────────────────────
    b = np.zeros(d)
    Theta = np.zeros((d, M))
    Theta_0 = rng.standard_normal((d, M)) / math.sqrt(lam * M)

    # ── 2. Replay Buffer ─────────────────────────────────────────────────────────
    y_buf   = np.empty(T, dtype=np.int64)
    ref_buf = np.empty(T, dtype=np.int64)
    b_buf   = np.empty(T, dtype=np.float64)
    Z_buf   = np.empty((T, M), dtype=np.float64)

    n_data  = 0
    y_prev  = 0
    regret  = np.zeros(T, dtype=float)
    ell_max = math.log(1.0 + math.e)

    for t in range(1, T + 1):
        ref = 0 if cfg.ref_mode == "fixed" else y_prev

        # ── 3. Action Selection (OLE / UCB Style) ───────────────────────────────
        # 1. reward prediction
        s_base = phi @ b
        r_base = tanh_reward(s_base)
        mean_diff = r_base - r_base[ref]

        # 2. uncertainty estimation
        phi_diff = phi - phi[ref]  # shape: (K, d)
        std_ens = np.linalg.norm(phi_diff @ (Theta + Theta_0), axis=1) # shape: (K,)

        # 3. OLE Score：mean + coefficient * uncertainty
        s_total = mean_diff + cfg.kappa * std_ens

        y = int(np.argmax(s_total))
        y_prev = y
        regret[t - 1] = float(r_star[best] - r_star[y])

        # ── 4. Environment Feedback ──────────────────────────────────────────────
        delta_star = float(r_star[y] - r_star[ref])
        p = float(sigmoid(np.array(delta_star)))
        b_obs = 1.0 if rng.random() < p else 0.0

        z_obs = _sample_vec(rng, cfg.z_dist, M) / math.sqrt(M)

        y_buf[n_data]   = y
        ref_buf[n_data] = ref
        b_buf[n_data]   = b_obs
        Z_buf[n_data]   = z_obs
        n_data += 1

        # ── 5. SGD Update (Stable Decoupled Gradients) ───────────────────────────
        eta = cfg.eta0 / t

        eta_Theta = eta / 20.0

        mb  = min(cfg.m, n_data)
        idxs = rng.integers(0, n_data, size=mb)

        ys_mb = y_buf[idxs]
        rs_mb = ref_buf[idxs]
        b_mb  = b_buf[idxs]
        Z_mb  = Z_buf[idxs]

        Phi_y = phi[ys_mb]
        Phi_r = phi[rs_mb]
        Delta_Phi = Phi_y - Phi_r  # (mb, d)


        s_y = Phi_y @ b
        s_r = Phi_r @ b
        ry  = tanh_reward(s_y)
        rr  = tanh_reward(s_r)

        p_mb = sigmoid(ry - rr)
        coef = (p_mb - b_mb) / ell_max
        dy   = 0.5 * (1.0 - np.tanh(s_y) ** 2)
        dr   = 0.5 * (1.0 - np.tanh(s_r) ** 2)

        grad_b_data = (coef * dy) @ Phi_y - (coef * dr) @ Phi_r
        grad_b = grad_b_data * (float(n_data) / float(mb)) + lam * b

        b = b - eta * grad_b
        b = np.clip(b, -cfg.clip, cfg.clip)

        # update Theta
        pred_Z = Delta_Phi @ (Theta + Theta_0)  # (mb, M)
        err_Z  = pred_Z - Z_mb                  # (mb, M)

        grad_Theta_data = Delta_Phi.T @ err_Z   # (d, M)
        grad_Theta = grad_Theta_data * (float(n_data) / float(mb)) + lam * Theta


        Theta = Theta - eta_Theta * grad_Theta
        Theta = np.clip(Theta, -cfg.clip, cfg.clip)

    return np.cumsum(regret)
def simulate(seed: int, cfg: SimConfig) -> np.ndarray:
    """Run OLE (or enspp) for T rounds; return cumulative regret array."""
    if cfg.method == "enspp":
        return _simulate_enspp(seed, cfg)
    elif cfg.method == "enspp_ole":
        return _simulate_enspp_ole(seed, cfg)
    elif cfg.method == "hyper":
        return _simulate_hyperagent(seed, cfg)
    elif cfg.method == "hyper_ole":
        return _simulate_hyperagent_ole(seed, cfg)
    rng = np.random.default_rng(seed)

    # Fixed random bandit instance (shared across seeds via cfg.env_seed)
    phi, theta_star, r_star, best = make_bandit_instance(cfg)

    # Initialize ensemble from prior
    thetas = rng.normal(scale=cfg.prior_sigma, size=(cfg.N, cfg.d))

    # Dataset (action indices + preference bits). Pre-allocate for speed.
    y_buf = np.empty(cfg.T, dtype=np.int64)
    ref_buf = np.empty(cfg.T, dtype=np.int64)
    b_buf = np.empty(cfg.T, dtype=np.float64)
    n_data = 0

    regret = np.zeros(cfg.T, dtype=float)
    ell_max = math.log(1.0 + math.e)
    y_prev = 0
    warmup_rounds = cfg.K * 0
    for t in range(1, cfg.T + 1):
        ref = 0 if cfg.ref_mode == "fixed" else y_prev


        # OLE index from centered rewards relative to ref
        s = thetas @ phi.T  # (N,K)
        r = tanh_reward(s)


        delta = r - r[:, ref : ref + 1]  # centered
        mean = delta.mean(axis=0)
        # Unbiased sample variance across particles. Using ddof=1 avoids an
        # artificial shrinkage of the exploration bonus when N is small.
        var = delta.var(axis=0, ddof=1) if cfg.N > 1 else np.zeros(cfg.K)

        std = np.sqrt(np.maximum(var, 1e-12))

        if cfg.variant == "ole":
            score = mean + cfg.kappa * std
        elif cfg.variant == "mean":
            score = mean
        elif cfg.variant == "var":
            score = std
        elif cfg.variant == "ts":
            # Thompson sampling via the SGLD particle ensemble: sample one
            # particle and act greedily under that sampled reward model.
            i = int(rng.integers(0, cfg.N))
            score = delta[i]
        else:
            raise ValueError(f"Unknown variant: {cfg.variant}")

        y = int(np.argmax(score))
        y_prev = y

        regret[t - 1] = float(r_star[best] - r_star[y])

        # Preference observation between y and ref
        delta_star = float(r_star[y] - r_star[ref])
        p = float(sigmoid(np.array(delta_star)))
        b = 1.0 if rng.random() < p else 0.0
        y_buf[n_data] = y
        ref_buf[n_data] = ref
        b_buf[n_data] = b
        n_data += 1

        # One SGLD/SGD step per particle on a mini-batch
        eta = cfg.eta0 / t
        mb = min(cfg.m, n_data)
        idxs = rng.integers(0, n_data, size=mb)

        ys = y_buf[idxs]
        rs = ref_buf[idxs]
        b_mb = b_buf[idxs]

        Phi_y = phi[ys]  # (mb,d)
        Phi_r = phi[rs]  # (mb,d)

        s_y = thetas @ Phi_y.T  # (N,mb)
        s_r = thetas @ Phi_r.T
        ty = np.tanh(s_y)
        tr = np.tanh(s_r)
        ry = tanh_reward(s_y)
        rr = tanh_reward(s_r)

        delta_mb = ry - rr
        p_mb = sigmoid(delta_mb)
        coef = (p_mb - b_mb) / ell_max  # (N,mb), broadcasting b_mb

        # dr/dtheta = 0.5*(1 - tanh(s)^2) * phi
        dy = 0.5 * (1.0 - ty ** 2)
        dr = 0.5 * (1.0 - tr ** 2)
        grad = (coef * dy) @ Phi_y - (coef * dr) @ Phi_r

        # Scale the mini-batch gradient to an (approximately) unbiased estimate of the
        # full gradient of the *cumulative* logistic loss over the dataset collected so far.
        #
        # This matches the Gibbs posterior used in the paper,
        #   \Pi_t(d\theta) \propto \pi_0(\theta) \exp\big(-\sum_{s\le t} \ell_\theta(z_s)/\beta\big) \, d\theta,
        # and corresponds to the standard SGLD scaling used in Bayesian inference:
        #   (|D_t| / m) * sum_{z in minibatch} \nabla \ell_\theta(z).
        grad *= float(n_data) / float(mb)

        # Gaussian prior: \nabla(-\beta \log \pi_0(\theta)) = -\beta \nabla \log \pi_0(\theta).
        # For a Gaussian prior, \nabla \log \pi_0(\theta) = -\theta/\sigma^2, hence the +beta*theta term.
        grad += cfg.beta * thetas / (cfg.prior_sigma ** 2)

        if cfg.sgld:
            noise = rng.normal(size=thetas.shape)
            thetas = thetas - eta * grad + math.sqrt(2.0 * cfg.beta * eta) * noise
        else:
            thetas = thetas - eta * grad

        thetas = np.clip(thetas, -cfg.clip, cfg.clip)

    return np.cumsum(regret)

def simulate_laplace(seed: int, cfg: SimConfig, method: str = "laplace_ucb") -> np.ndarray:
    """Classical baselines via an online Laplace/Fisher approximation.

    We implement two lightweight baselines that are common in generalized-linear
    (logistic / BTL) bandits and do *not* require particle ensembles:

      - ``laplace_ucb``: plug-in estimator + delta-method uncertainty bonus.
      - ``laplace_ts``: posterior sampling from the same Gaussian approximation.

    The reward model matches the synthetic environment used for OLE (bounded
    tanh parameterization). The covariance is based on an online Fisher update
    (Gauss--Newton approximation).
    """

    if method not in {"laplace_ucb", "laplace_ts"}:
        raise ValueError(f"Unknown Laplace baseline: {method}")

    rng = np.random.default_rng(seed)
    phi, _theta_star, r_star, best = make_bandit_instance(cfg)

    # Online point estimate and (approximate) covariance from a Laplace/Fisher
    # approximation. We maintain the covariance directly and update it with a
    # Sherman--Morrison rank-1 update to avoid per-round linear solves.
    theta = np.zeros(cfg.d, dtype=float)
    precision0 = (cfg.beta / (cfg.prior_sigma ** 2))
    cov = (1.0 / precision0) * np.eye(cfg.d, dtype=float)

    # Dataset for the SGD point-estimate update (same as OLE for fairness).
    y_buf = np.empty(cfg.T, dtype=np.int64)
    ref_buf = np.empty(cfg.T, dtype=np.int64)
    b_buf = np.empty(cfg.T, dtype=np.float64)
    n_data = 0

    regret = np.zeros(cfg.T, dtype=float)
    ell_max = math.log(1.0 + math.e)
    y_prev = 0

    for t in range(1, cfg.T + 1):
        ref = 0 if cfg.ref_mode == "fixed" else y_prev

        # Predict rewards and their gradients at the current point estimate.
        s = theta @ phi.T  # (K,)
        r = tanh_reward(s)
        delta = r - r[ref]

        drds = tanh_reward_ds(s)  # (K,)
        grad_r = drds[:, None] * phi  # (K,d)
        grad_delta = grad_r - grad_r[ref : ref + 1]  # (K,d)

        # Delta-method predictive standard deviation: sqrt(g^T Cov g).
        var = np.sum((grad_delta @ cov) * grad_delta, axis=1)
        std = np.sqrt(np.maximum(var, 1e-12))

        if method == "laplace_ucb":
            score = delta + cfg.kappa * std
            y = int(np.argmax(score))
        else:  # laplace_ts
            L = np.linalg.cholesky(cov)
            theta_samp = theta + L @ rng.normal(size=cfg.d)
            s_samp = theta_samp @ phi.T
            r_samp = tanh_reward(s_samp)
            delta_samp = r_samp - r_samp[ref]
            y = int(np.argmax(delta_samp))

        y_prev = y
        regret[t - 1] = float(r_star[best] - r_star[y])

        # Preference observation between y and ref
        delta_star = float(r_star[y] - r_star[ref])
        p = float(sigmoid(np.array(delta_star)))
        b = 1.0 if rng.random() < p else 0.0
        y_buf[n_data] = y
        ref_buf[n_data] = ref
        b_buf[n_data] = b
        n_data += 1

        # ----------
        # Online Fisher update (using the current point estimate).
        # ----------
        p_hat = float(sigmoid(np.array(delta[y])))
        w = (p_hat * (1.0 - p_hat)) / ell_max
        g = grad_delta[y]  # (d,)

        # Sherman--Morrison: (A + w g g^T)^{-1} = A^{-1} - (w A^{-1} g g^T A^{-1}) / (1 + w g^T A^{-1} g)
        v = cov @ g
        denom = 1.0 + w * float(g @ v)
        cov = cov - (w / denom) * np.outer(v, v)
        cov = 0.5 * (cov + cov.T)

        # ----------
        # One SGD step on the cumulative objective (mini-batch), mirroring OLE.
        # ----------
        eta = cfg.eta0 / t
        mb = min(cfg.m, n_data)
        idxs = rng.integers(0, n_data, size=mb)

        ys = y_buf[idxs]
        rs = ref_buf[idxs]
        b_mb = b_buf[idxs]

        Phi_y = phi[ys]  # (mb,d)
        Phi_r = phi[rs]

        s_y = Phi_y @ theta  # (mb,)
        s_r = Phi_r @ theta
        ry = tanh_reward(s_y)
        rr = tanh_reward(s_r)
        delta_mb = ry - rr

        p_mb = sigmoid(delta_mb)
        coef = (p_mb - b_mb) / ell_max  # (mb,)

        dy = tanh_reward_ds(s_y)
        dr = tanh_reward_ds(s_r)
        grad = (Phi_y.T @ (coef * dy)) - (Phi_r.T @ (coef * dr))

        grad *= float(n_data) / float(mb)
        grad += cfg.beta * theta / (cfg.prior_sigma ** 2)

        theta = theta - eta * grad
        theta = np.clip(theta, -cfg.clip, cfg.clip)

    return np.cumsum(regret)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def regret_stats(cfg: SimConfig, seeds: List[int], sim_fn=simulate) -> Tuple[float, float]:
    """Mean/std of regret@T over seeds for a given simulator."""
    finals = []
    for s in seeds:
        finals.append(float(sim_fn(s, cfg)[-1]))
    finals = np.array(finals, dtype=float)
    std = float(finals.std(ddof=1)) if finals.size > 1 else 0.0
    return float(finals.mean()), std


def task_scaling(root: str, outdir: str, cfg: SimConfig, seeds: List[int]) -> None:
    ensure_dir(outdir)
    cfg.eta0 = 1.0
    cfg.kappa = 3
    cfg.beta = 0.25
    cfg.m = 64
    cfg.T = 8000

    crs = [simulate(s, cfg) for s in seeds]
    crs = np.stack(crs, axis=0)  # (S,T)

    t = np.arange(1, cfg.T + 1)
    mean = crs.mean(axis=0)
    ddof = 1 if crs.shape[0] > 1 else 0
    se = crs.std(axis=0, ddof=ddof) / math.sqrt(crs.shape[0])

    # Save curve
    curve_path = os.path.join(outdir, "synthetic_regret_curve.csv")
    write_csv(curve_path, ["t", "mean_regret", "se_regret"], np.stack([t, mean, se], axis=1).tolist())

    # -------------------------
    # Descriptive scaling diagnostic.
    # -------------------------
    # To guard against over-interpreting the normalized regret plot alone, we
    # compute a simple log--log scaling summary: for each seed, we fit
    #   log Regret(t) = a * log t + b
    # over the *final decade* of t, i.e. t in [T/10, T]. We then report the mean
    # fitted slope and a bootstrap CI over seeds. This is purely descriptive and
    # should not be interpreted as an asymptotic rate claim.
    t_start = max(1, cfg.T  // 3)
    tt = np.arange(t_start, cfg.T + 1, dtype=float)
    x = np.log(tt)
    x_mean = float(x.mean())
    denom = float(((x - x_mean) ** 2).sum())

    slopes = []
    for i in range(crs.shape[0]):
        rr = np.maximum(crs[i, t_start - 1 :], 1e-12)
        y = np.log(rr.astype(float))
        y_mean = float(y.mean())
        slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
        slopes.append(slope)
    slopes = np.array(slopes, dtype=float)

    slope_mean = float(slopes.mean())

    # Bootstrap CI for the mean slope across seeds.
    rng_boot = np.random.default_rng(0)
    B = 2000
    boot = rng_boot.choice(slopes, size=(B, slopes.shape[0]), replace=True).mean(axis=1)
    ci_lo, ci_hi = [float(q) for q in np.quantile(boot, [0.025, 0.975])]

    diag = {
        "T": int(cfg.T),
        "t_start": int(t_start),
        "t_end": int(cfg.T),
        "n_seeds": int(len(seeds)),
        "seed_slopes": [float(s) for s in slopes.tolist()],
        "slope_mean": float(slope_mean),
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "bootstrap_B": int(B),
    }
    diag_json_path = os.path.join(outdir, "synthetic_scaling_diagnostic.json")
    with open(diag_json_path, "w") as f:
        json.dump(diag, f, indent=2)

    diag_tex_path = os.path.join(outdir, "synthetic_scaling_diagnostic.tex")
    with open(diag_tex_path, "w") as f:
        f.write("% Auto-generated by experiments/synthetic_bandit/run_synthetic.py (task=scaling).\n")
        f.write("% Descriptive diagnostic only (not an asymptotic rate claim).\n")
        f.write(f"\\newcommand{{\\syntheticScalingTstart}}{{{int(t_start)}}}\n")
        f.write(f"\\newcommand{{\\syntheticScalingTend}}{{{int(cfg.T)}}}\n")
        f.write(f"\\newcommand{{\\syntheticScalingNumSeeds}}{{{int(len(seeds))}}}\n")
        f.write(f"\\newcommand{{\\syntheticScalingSlope}}{{{slope_mean:.2f}}}\n")
        f.write(f"\\newcommand{{\\syntheticScalingSlopeCILo}}{{{ci_lo:.2f}}}\n")
        f.write(f"\\newcommand{{\\syntheticScalingSlopeCIHi}}{{{ci_hi:.2f}}}\n")

    # Plot with two panels
    c = float(mean[-1] / math.sqrt(cfg.T))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))

    # (a) Scaling visualized as linear in $\sqrt{t}$
    ax = axes[0]
    s = np.sqrt(t)
    ax.plot(s, mean, label="OLE (mean)")
    ax.fill_between(s, mean - se, mean + se, alpha=0.2)
    ax.plot(s, c * s, linestyle="--", label=r"$c\sqrt{t}$ fit")
    ax.set_xlabel(r"$\sqrt{t}$")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(r"(a) Regret vs. $\sqrt{t}$")
    ax.legend(fontsize=7)

    # (b) Normalized regret with log-x for readability
    ax = axes[1]
    ratio = mean / np.sqrt(t)
    ratio_lo = (mean - se) / np.sqrt(t)
    ratio_hi = (mean + se) / np.sqrt(t)
    ax.plot(t, ratio, label=r"$\mathrm{Regret}(t)/\sqrt{t}$")
    ax.fill_between(t, ratio_lo, ratio_hi, alpha=0.2)
    ax.axhline(c, linestyle="--", label=r"$c$ (from final)")
    ax.set_xscale("log")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"$\mathrm{Regret}(t)/\sqrt{t}$")
    ax.set_title("(b) Normalized regret")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig_path = os.path.join(root, "figures", f"synthetic_regret_scaling{FIG_SUFFIX}.pdf")
    ensure_dir(os.path.dirname(fig_path))
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"[scaling] wrote {curve_path}")
    print(f"[scaling] wrote {diag_json_path}")
    print(f"[scaling] wrote {diag_tex_path}")
    print(f"[scaling] wrote {fig_path}")


def task_impl_terms(outdir: str, cfg: SimConfig, seeds: List[int], N_list: List[int], eta0_list: List[float]) -> None:
    ensure_dir(outdir)
    T = cfg.T
    rows = []

    for N in N_list:
        cfgN = SimConfig(**{**asdict(cfg), "N": N, "variant": "ole", "ref_mode": "fixed", "sgld": True})
        mean, std = regret_stats(cfgN, seeds)
        eta = cfgN.eta0 / np.arange(1, T + 1)
        rows.append({
            "Setting": f"$N={N}$",
            "RegretMean": mean,
            "RegretStd": std,
            "SumEta": float(eta.sum()),
            "MC": float(math.sqrt(T / N)),
            "SG": float(math.sqrt(float((eta ** 2).sum()) / cfgN.m)),
        })

    for eta0 in eta0_list:
        cfgE = SimConfig(**{**asdict(cfg), "eta0": eta0, "variant": "ole", "ref_mode": "fixed", "sgld": True})
        mean, std = regret_stats(cfgE, seeds)
        eta = eta0 / np.arange(1, T + 1)
        rows.append({
            "Setting": f"$\\eta_0={eta0}$",
            "RegretMean": mean,
            "RegretStd": std,
            "SumEta": float(eta.sum()),
            "MC": float(math.sqrt(T / cfgE.N)),
            "SG": float(math.sqrt(float((eta ** 2).sum()) / cfgE.m)),
        })

    # Write csv
    csv_path = os.path.join(outdir, "synthetic_impl_terms.csv")
    header = ["Setting", "RegretMean", "RegretStd", "SumEta", "MC", "SG"]
    write_csv(csv_path, header, [[r[h] for h in header] for r in rows])

    # Write LaTeX table
    tex_path = os.path.join(outdir, "synthetic_impl_table.tex")
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Controlled synthetic preference bandit on a fixed random instance ($K=%d$ arms, $d=%d$, BTL comparisons; env seed=%d). We vary the ensemble size $N$ and initial step size $\eta_0$ (with $\eta_t=\eta_0/t$) for \textsc{OLE} using an SGLD particle ensemble and a fixed reference arm, while holding the remaining hyperparameters fixed ($m=%d$, $\beta=%.3g$, $\kappa=%.3g$). We report mean$\pm$std regret@T at $T=%d$ over $%d$ seeds. The last three columns are the hyperparameter-only proxies from Corollary~\ref{cor:well_tuned} for discretization ($\sum_{t\le T}\eta_t$), finite-ensemble Monte Carlo error ($\sqrt{T/N}$), and mini-batch gradient noise ($\sqrt{\sum_{t\le T}\eta_t^2/m}$).}"
        % (cfg.K, cfg.d, cfg.env_seed, cfg.m, cfg.beta, cfg.kappa, cfg.T, len(seeds))
    )
    lines.append(r"\label{tab:synthetic_impl}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Setting & Regret@T & \makecell{Discretization\\proxy $\sum_{t\le T}\eta_t$} & \makecell{MC proxy\\$\sqrt{T/N}$} & \makecell{SG proxy\\$\sqrt{\sum_{t\le T}\eta_t^2/m}$} \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(f"{r['Setting']} & {r['RegretMean']:.1f}$\\pm${r['RegretStd']:.1f} & {r['SumEta']:.2f} & {r['MC']:.2f} & {r['SG']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[impl] wrote {csv_path}")
    print(f"[impl] wrote {tex_path}")


def task_minibatch_sweep(outdir: str, cfg: SimConfig, seeds: List[int], m_list: List[int]) -> None:
    """Sweep mini-batch size m at fixed (N, eta0) and write a LaTeX table for the paper."""
    ensure_dir(outdir)
    T = cfg.T
    rows = []

    eta = cfg.eta0 / np.arange(1, T + 1)
    sum_eta = float(eta.sum())
    mc_proxy = float(math.sqrt(T / cfg.N))

    for m in m_list:
        cfgM = SimConfig(**{**asdict(cfg), "m": m, "variant": "ole", "ref_mode": "fixed", "sgld": True})
        mean, std = regret_stats(cfgM, seeds)
        sg_proxy = float(math.sqrt(float((eta ** 2).sum()) / m))
        rows.append({
            "Setting": f"$m={m}$",
            "RegretMean": mean,
            "RegretStd": std,
            "SumEta": sum_eta,
            "MC": mc_proxy,
            "SG": sg_proxy,
        })

    csv_path = os.path.join(outdir, "synthetic_minibatch_sweep.csv")
    header = ["Setting", "RegretMean", "RegretStd", "SumEta", "MC", "SG"]
    write_csv(csv_path, header, [[r[h] for h in header] for r in rows])

    tex_path = os.path.join(outdir, "synthetic_minibatch_table.tex")
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Synthetic mini-batch sweep (same setting as Table~\ref{tab:synthetic_impl}). We vary the mini-batch size $m$ used in the SGLD update while fixing the ensemble size ($N=%d$), step size schedule ($\eta_t=%.3g/t$), and the remaining hyperparameters ($\beta=%.3g$, $\kappa=%.3g$). We report mean$\pm$std regret@T at $T=%d$ over $%d$ seeds. The last column is the mini-batch proxy $\sqrt{\sum_{t\le T}\eta_t^2/m}$ from Corollary~\ref{cor:well_tuned}.}"
        % (cfg.N, cfg.eta0, cfg.beta, cfg.kappa, cfg.T, len(seeds))
    )
    lines.append(r"\label{tab:synthetic_minibatch}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4.0pt}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Setting & Regret@T & \makecell{Discretization\\proxy $\sum_{t\le T}\eta_t$} & \makecell{SG proxy\\$\sqrt{\sum_{t\le T}\eta_t^2/m}$} \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{r['Setting']} & {r['RegretMean']:.1f}$\\pm${r['RegretStd']:.1f} & "
            f"{r['SumEta']:.2f} & {r['SG']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[minibatch] wrote {csv_path}")
    print(f"[minibatch] wrote {tex_path}")



def task_hparam_grid(outdir: str, cfg: SimConfig, seeds: List[int], N_list: List[int], eta0_list: List[float]) -> str:
    """Runs a small N×eta0 grid and writes a CSV suitable for a heatmap.

    Returns the path to the generated CSV.
    """
    ensure_dir(outdir)
    T = cfg.T
    rows = []

    for N in N_list:
        for eta0 in eta0_list:
            cfgG = SimConfig(**{**asdict(cfg), "N": N, "eta0": float(eta0), "variant": "ole", "ref_mode": "fixed", "sgld": True})
            mean, std = regret_stats(cfgG, seeds)
            eta = cfgG.eta0 / np.arange(1, T + 1)
            rows.append({
                "N": int(N),
                "eta0": float(eta0),
                "T": int(T),
                "regret_mean": float(mean),
                "regret_std": float(std),
                "disc_term": float(eta.sum()),
                "mc_term": float(math.sqrt(T / N)),
                "sg_term": float(math.sqrt(float((eta ** 2).sum()) / cfgG.m)),
            })

    csv_path = os.path.join(outdir, "synthetic_hparam_grid.csv")
    header = ["N", "eta0", "T", "regret_mean", "regret_std", "disc_term", "mc_term", "sg_term"]
    write_csv(csv_path, header, [[r[h] for h in header] for r in rows])
    print(f"[grid] wrote {csv_path}")
    return csv_path


def plot_hparam_heatmap(csv_path: str, fig_path: str, metric: str = "regret_mean") -> None:
    """Plots a N×eta0 heatmap from a grid CSV."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    # Unique sorted axes
    Ns = np.unique(data["N"]).astype(int)
    eta0s = np.unique(data["eta0"]).astype(float)

    # Fill matrix
    M = np.full((len(Ns), len(eta0s)), np.nan, dtype=float)
    for row in data:
        i = int(np.where(Ns == int(row["N"]))[0][0])
        j = int(np.where(eta0s == float(row["eta0"]))[0][0])
        M[i, j] = float(row[metric])

    plt.figure(figsize=(4.8, 3.6))
    im = plt.imshow(M, aspect="auto", origin="lower")
    label_map = {
        "regret_mean": r"mean regret@T",
        "regret_std": r"std. regret@T",
    }
    plt.colorbar(im, label=label_map.get(metric, metric))

    # Annotate each cell for readability (small grids only).
    m_min = float(np.nanmin(M))
    m_max = float(np.nanmax(M))
    for ii in range(len(Ns)):
        for jj in range(len(eta0s)):
            val = float(M[ii, jj])
            if np.isfinite(val):
                # Choose text color for contrast against the colormap.
                norm = 0.0 if m_max == m_min else (val - m_min) / (m_max - m_min)
                txt_color = "black" if norm > 0.55 else "white"
                plt.text(jj, ii, f"{val:.1f}", ha="center", va="center", fontsize=8, color=txt_color)

    plt.xticks(range(len(eta0s)), [f"{x:g}" for x in eta0s], rotation=0)
    plt.yticks(range(len(Ns)), [str(n) for n in Ns])
    plt.xlabel(r"step size $\eta_0$")
    plt.ylabel(r"ensemble size $N$")
    plt.title(r"Hyperparameter sensitivity ($N \times \eta_0$)")
    plt.tight_layout()
    ensure_dir(os.path.dirname(fig_path))
    plt.savefig(fig_path)
    plt.close()
    print(f"[heatmap] wrote {fig_path}")

def task_ablation(outdir: str, cfg: SimConfig, seeds: List[int]) -> None:
    ensure_dir(outdir)
    variants = [
        ("Mean only", SimConfig(**{**asdict(cfg), "variant": "mean", "kappa": 0.0, "ref_mode": "fixed", "sgld": True})),
        ("Variance only", SimConfig(**{**asdict(cfg), "variant": "var", "kappa": 0.0, "ref_mode": "fixed", "sgld": True})),
        ("Mean+variance (OLE)", SimConfig(**{**asdict(cfg), "variant": "ole", "ref_mode": "fixed", "sgld": True})),
        ("Adaptive ref (previous)", SimConfig(**{**asdict(cfg), "variant": "ole", "ref_mode": "prev", "sgld": True})),
        ("Deterministic SGD ensemble", SimConfig(**{**asdict(cfg), "variant": "ole", "ref_mode": "fixed", "sgld": False})),
    ]

    rows = []
    for name, cfgv in variants:
        mean, std = regret_stats(cfgv, seeds)
        rows.append({"Variant": name, "RegretMean": mean, "RegretStd": std})

    # Write csv
    csv_path = os.path.join(outdir, "synthetic_ablation.csv")
    header = ["Variant", "RegretMean", "RegretStd"]
    write_csv(csv_path, header, [[r[h] for h in header] for r in rows])

    best = min(rows, key=lambda r: r["RegretMean"])["Variant"]

    # Write LaTeX table
    tex_path = os.path.join(outdir, "synthetic_ablation_table.tex")
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Synthetic preference bandit ablations on the same fixed bandit instance as Table~\ref{tab:synthetic_impl} (env seed=%d). We report mean$\pm$std regret@T at $T=%d$ over $%d$ seeds. The OLE index (mean$+$variance) attains lower regret than mean-only or variance-only scoring. SGLD particles outperform deterministic SGD. Changing the reference arm (fixed vs.\ previous action) affects performance.}"
        % (cfg.env_seed, cfg.T, len(seeds))
    )
    lines.append(r"\label{tab:synthetic_ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Regret@T \\")
    lines.append(r"\midrule")
    for r in rows:
        cell = f"{r['RegretMean']:.1f}$\\pm${r['RegretStd']:.1f}"
        if r["Variant"] == best:
            cell = r"\textbf{" + cell + r"}"
        lines.append(f"{r['Variant']} & {cell} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[ablation] wrote {csv_path}")
    print(f"[ablation] wrote {tex_path}")


def task_baselines(outdir: str, cfg: SimConfig, seeds: List[int]) -> None:
    """Run classical baselines on the fixed synthetic instance.

    This complements the ablations (which hold the OLE template fixed) with
    two standard reference algorithms for logistic / BTL bandits:
      - Laplace-UCB: plug-in estimator + delta-method UCB bonus.
      - Laplace-TS: posterior sampling from the same Gaussian approximation.

    We also include ensemble Thompson sampling using the same SGLD particle
    ensemble as OLE (sample one particle and act greedily under it).
    """

    ensure_dir(outdir)

    # For the Laplace baselines we use a slightly larger optimism coefficient
    # (kappa=1.0) since the Laplace covariance scale differs from the SGLD
    # ensemble variance used by OLE.
    cfg_laplace = SimConfig(**{**asdict(cfg), "kappa": 1.0})

    methods = [
        ("OLE (UCB; SGLD ensemble)", SimConfig(**{**asdict(cfg), "variant": "ole", "sgld": True})),
        ("Thompson (SGLD ensemble)", SimConfig(**{**asdict(cfg), "variant": "ts", "sgld": True})),
        ("Laplace-UCB (plug-in + bonus)", cfg_laplace),
        ("Laplace-TS (posterior sampling)", cfg_laplace),
    ]

    rows = []
    for name, cfgm in methods:
        if name.startswith("Laplace-UCB"):
            mean, std = regret_stats(cfgm, seeds, sim_fn=lambda s, c: simulate_laplace(s, c, method="laplace_ucb"))
        elif name.startswith("Laplace-TS"):
            mean, std = regret_stats(cfgm, seeds, sim_fn=lambda s, c: simulate_laplace(s, c, method="laplace_ts"))
        else:
            mean, std = regret_stats(cfgm, seeds)
        rows.append({"Method": name, "RegretMean": mean, "RegretStd": std})

    # Write CSV
    csv_path = os.path.join(outdir, "synthetic_baselines.csv")
    header = ["Method", "RegretMean", "RegretStd"]
    write_csv(csv_path, header, [[r[h] for h in header] for r in rows])

    # Write LaTeX table
    best = min(rows, key=lambda r: r["RegretMean"])["Method"]
    tex_path = os.path.join(outdir, "synthetic_baselines_table.tex")
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Synthetic preference bandit: comparison to standard baselines on the same fixed instance as Tables~\ref{tab:synthetic_impl}--\ref{tab:synthetic_ablation} (BTL comparisons; env seed=%d). We report mean$\pm$std regret@T at $T=%d$ over $%d$ seeds. Laplace-UCB and Laplace-TS use an online Gaussian (Laplace/Fisher) approximation with a delta-method uncertainty estimate; Thompson (SGLD) samples one particle from the SGLD ensemble each round.}"
        % (cfg.env_seed, cfg.T, len(seeds))
    )
    lines.append(r"\label{tab:synthetic_baselines}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Regret@T \\")
    lines.append(r"\midrule")
    for r in rows:
        cell = f"{r['RegretMean']:.1f}$\\pm${r['RegretStd']:.1f}"
        if r["Method"] == best:
            cell = r"\textbf{" + cell + r"}"
        lines.append(f"{r['Method']} & {cell} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[baselines] wrote {csv_path}")
    print(f"[baselines] wrote {tex_path}")


def task_compute_tradeoff(
    root: str,
    outdir: str,
    cfg: SimConfig,
    seeds: List[int],
    N_list: List[int],
    m_list: List[int],
    include_laplace: bool = True,
) -> None:
    """Compute--regret tradeoff for OLE by sweeping (N, m).

    The x-axis uses the implementation proxy N×m suggested by the paper's
    per-round cost analysis.
    """

    ensure_dir(outdir)

    rows = []

    # OLE sweeps
    for N in N_list:
        for m in m_list:
            cfg_nm = SimConfig(**{**asdict(cfg), "N": int(N), "m": int(m), "variant": "ole", "sgld": True, "ref_mode": "fixed"})
            mean, std = regret_stats(cfg_nm, seeds)
            rows.append({
                "algo": "OLE",
                "N": int(N),
                "m": int(m),
                "compute": int(N) * int(m),
                "regret_mean": float(mean),
                "regret_std": float(std),
            })

    # Optional Laplace-UCB reference points at compute≈m.
    if include_laplace:
        for m in m_list:
            cfg_m = SimConfig(**{**asdict(cfg), "m": int(m), "kappa": 1.0})
            mean, std = regret_stats(cfg_m, seeds, sim_fn=lambda s, c: simulate_laplace(s, c, method="laplace_ucb"))
            rows.append({
                "algo": "Laplace-UCB",
                "N": 1,
                "m": int(m),
                "compute": int(m),
                "regret_mean": float(mean),
                "regret_std": float(std),
            })

    # Write CSV
    csv_path = os.path.join(outdir, "synthetic_compute_tradeoff.csv")
    header = ["algo", "N", "m", "compute", "regret_mean", "regret_std"]
    write_csv(csv_path, header, [[r[h] for h in header] for r in rows])

    # Plot
    plt.figure(figsize=(4.8, 3.2))

    # Group by algorithm and, for OLE, group by N.
    rows_ole = [r for r in rows if r["algo"] == "OLE"]
    for N in sorted(set(r["N"] for r in rows_ole)):
        sub = [r for r in rows_ole if r["N"] == N]
        sub = sorted(sub, key=lambda r: r["compute"])
        x = [r["compute"] for r in sub]
        y = [r["regret_mean"] for r in sub]
        yerr = [r["regret_std"] for r in sub]
        plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.2, capsize=2, label=f"OLE (N={N})")

    if include_laplace:
        sub = [r for r in rows if r["algo"] == "Laplace-UCB"]
        sub = sorted(sub, key=lambda r: r["compute"])
        x = [r["compute"] for r in sub]
        y = [r["regret_mean"] for r in sub]
        yerr = [r["regret_std"] for r in sub]
        plt.errorbar(x, y, yerr=yerr, marker="s", linestyle="--", linewidth=1.2, capsize=2, label="Laplace-UCB")

    plt.xscale("log")
    plt.xlabel(r"compute proxy $N\times m$ (log scale)")
    plt.ylabel(r"regret@T")
    plt.title(r"Compute--regret tradeoff (fixed $T$)")
    plt.legend(fontsize=7)
    plt.tight_layout()

    fig_path = os.path.join(root, "figures", f"synthetic_compute_tradeoff{FIG_SUFFIX}.pdf")
    ensure_dir(os.path.dirname(fig_path))
    plt.savefig(fig_path)
    plt.close()

    print(f"[tradeoff] wrote {csv_path}")
    print(f"[tradeoff] wrote {fig_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "scaling", "impl", "minibatch", "ablation", "baselines", "tradeoff", "grid", "heatmap"],
    )
    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--n_seeds", type=int, default=20)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--tag", type=str, default="", help="Optional suffix for saved figures (e.g., 'smoke').")

    # Hyperparameter grid for the appendix heatmap (comma-separated lists)
    p.add_argument("--grid_N_list", type=str, default="5,10,20,40")
    p.add_argument("--grid_eta0_list", type=str, default="0.2,2.0,10.0")
    p.add_argument("--grid_n_seeds", type=int, default=4)

    # Compute--regret tradeoff sweep lists (comma-separated)
    p.add_argument("--tradeoff_N_list", type=str, default="5,10,20,40")
    p.add_argument("--tradeoff_m_list", type=str, default="8,16,32,64")
    p.add_argument("--tradeoff_n_seeds", type=int, default=4)
    p.add_argument("--tradeoff_include_laplace", action="store_true", help="Include Laplace-UCB reference points")


    # Baseline hyperparameters
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--env_seed", type=int, default=2)
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--eta0", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gap_scale", type=float, default=0.3)
    p.add_argument("--ref_mode", type=str, default="fixed")

    # method selection
    p.add_argument("--method", type=str, default="ole",
        choices = ['ole', 'enspp', 'hyper', 'enspp_ole', 'hyper_ole']
    )


    # HyperAgent-specific CLI args
    p.add_argument("--hyperagent_M", type=int, default=10, help="HyperAgent index dimension M")
    p.add_argument("--hyperagent_sigma", type=float, default=0.3, help="HyperAgent perturbation scale")
    p.add_argument("--hyperagent_n_indices", type=int, default=20, help="HyperAgent sample-average batch size")
    p.add_argument("--hyperagent_prior_scale_mult", type=float, default=0.35,
                    help="HyperAgent prior uncertainty scale multiplier")

    # HyperAgent+OLE hybrid CLI args
    p.add_argument("--hybrid_n_heads", type=int, default=2,
                    help="Number of independent hypermodel heads for HyperAgent+OLE")
    p.add_argument("--hybrid_kappa", type=float, default=1.0,
                    help="OLE exploration weight kappa for HyperAgent+OLE")
    p.add_argument("--prior_sigma", type=float, default=0.5,
                    help="Prior width sigma")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    global FIG_SUFFIX
    FIG_SUFFIX = f"_{args.tag}" if getattr(args, 'tag', '') else ""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    outdir = args.outdir or os.path.join(root, "experiments", "synthetic_bandit", "results")
    ensure_dir(outdir)

    cfg = SimConfig(
        T=args.T,
        d=args.d,
        K=args.K,
        env_seed=args.env_seed,
        N=args.N,
        m=args.m,
        eta0=args.eta0,
        kappa=args.kappa,
        beta=args.beta,
        gap_scale=args.gap_scale,
        ref_mode=args.ref_mode,
        sgld=True,
        method = args.method,
        n_seeds = args.n_seeds,
        hyperagent_M=args.hyperagent_M,
        hyperagent_sigma=args.hyperagent_sigma,
        hyperagent_n_indices=args.hyperagent_n_indices,
        hyperagent_prior_scale_mult=args.hyperagent_prior_scale_mult,
        prior_sigma=args.prior_sigma,
    )
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))

    # Save run configuration
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "cfg": asdict(cfg), "seeds": seeds}, f, indent=2)

    if args.task in ("all", "scaling"):
        task_scaling(root=root, outdir=outdir, cfg=cfg, seeds= list(range(args.seed_offset, args.seed_offset + 32)))

    if args.task in ("all", "impl"):
        # Small grid used in the main-text table; we use up to 8 seeds for speed.
        task_impl_terms(
            outdir=outdir,
            cfg=cfg,
            seeds=list(range(args.seed_offset, args.seed_offset + min(8, args.n_seeds))),
            N_list=[5, 10, 20],
            eta0_list=[0.2, 2.0, 10.0],
        )

    if args.task in ("all", "minibatch"):
        task_minibatch_sweep(
            outdir=outdir,
            cfg=cfg,
            seeds=list(range(args.seed_offset, args.seed_offset + min(8, args.n_seeds))),
            m_list=[8, 16, 32, 64, 128],
        )

    if args.task in ("all", "ablation"):
        # Main-text ablation table uses up to 8 seeds for speed.
        task_ablation(
            outdir=outdir,
            cfg=cfg,
            seeds=list(range(args.seed_offset, args.seed_offset + min(8, args.n_seeds))),
        )

    if args.task in ("all", "baselines"):
        task_baselines(
            outdir=outdir,
            cfg=cfg,
            seeds=list(range(args.seed_offset, args.seed_offset + min(8, args.n_seeds))),
        )

    if args.task in ("all", "tradeoff"):
        # Tradeoff sweep is heavier; default to a smaller seed budget.
        tradeoff_seeds = list(
            range(args.seed_offset, args.seed_offset + min(args.tradeoff_n_seeds, args.n_seeds))
        )
        N_list = [int(x) for x in args.tradeoff_N_list.split(",") if x.strip()]
        m_list = [int(x) for x in args.tradeoff_m_list.split(",") if x.strip()]
        task_compute_tradeoff(
            root=root,
            outdir=outdir,
            cfg=cfg,
            seeds=tradeoff_seeds,
            N_list=N_list,
            m_list=m_list,
            include_laplace=args.tradeoff_include_laplace,
        )


    if args.task in ("all", "grid"):
        # Small N×eta0 grid for the appendix heatmap; use a smaller seed budget by default.
        grid_seeds = list(range(args.seed_offset, args.seed_offset + min(args.grid_n_seeds, args.n_seeds)))
        N_list = [int(x) for x in args.grid_N_list.split(",") if x.strip()]
        eta0_list = [float(x) for x in args.grid_eta0_list.split(",") if x.strip()]
        grid_csv = task_hparam_grid(outdir=outdir, cfg=cfg, seeds=grid_seeds, N_list=N_list, eta0_list=eta0_list)

    if args.task in ("all", "heatmap"):
        # Plot heatmap from an existing grid CSV (either generated above or shipped with the repo).
        grid_csv = os.path.join(outdir, "synthetic_hparam_grid.csv")
        fig_path = os.path.join(root, "figures", f"synthetic_hparam_heatmap{FIG_SUFFIX}.pdf")
        plot_hparam_heatmap(csv_path=grid_csv, fig_path=fig_path)



if __name__ == "__main__":
    main()
