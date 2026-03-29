#!/usr/bin/env python3
"""Lightweight RLHF-bridging experiment using SHP + frozen text features.

We build a small contextual preference-bandit environment from the Stanford
Human Preferences (SHP) dataset by grouping responses by prompt and defining a
fixed *reference response* per prompt. Each round:

  1) sample a prompt/context,
  2) choose one candidate response y from a small set Y(x),
  3) observe a noisy preference bit comparing y to the prompt-specific
     reference response \bar y(x),
  4) update an SGLD particle ensemble on the BTL/logistic likelihood.

The goal is to keep the experiment reproducible and small while remaining
conceptually aligned with RLHF (compare a proposed response to a baseline).

Default dependencies (TF--IDF encoder): numpy, scipy, scikit-learn, matplotlib.

Optional encoder: a frozen MiniLM/SBERT sentence encoder (no finetuning).
To use it, pass ``--encoder sbert`` and install ``transformers`` (and ``torch``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


FeatMat = Union[np.ndarray, sparse.csr_matrix]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


SHP_RAW_BASE = "https://huggingface.co/datasets/stanfordnlp/SHP/raw/main"
SHP_FILES = {
    "askphysics_validation": f"{SHP_RAW_BASE}/askphysics/validation.json",
    "askphysics_test": f"{SHP_RAW_BASE}/askphysics/test.json",
}


def download_if_needed(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    tmp = path + ".tmp"
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            f.write(r.read())
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {url}.\n"
            f"If you are running offline, manually download the file and place it at {path}."
        ) from e
    os.replace(tmp, path)


def load_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@dataclass
class Comment:
    cid: str
    text: str
    score: float


@dataclass
class PromptPool:
    """One context/prompt with a fixed reference response and K candidate actions."""

    post_id: str
    history: str
    baseline: Comment
    candidates: List[Comment]


@dataclass
class PreparedPool:
    """Pool with precomputed feature diffs relative to baseline.

    ``X_diff`` can be either sparse (CSR) for TF--IDF, or dense (ndarray)
    for transformer-based embeddings.
    """

    X_diff: FeatMat  # (K,d)
    u_base: float
    u_cands: np.ndarray
    u_best: float


def build_pools(records: List[dict], *, k_total: int = 6, seed: int = 0) -> List[PromptPool]:
    """Group SHP records by post_id and build multi-action pools.

    We select a small set of candidate responses per prompt by mixing high-score
    and low-score comments. The baseline (reference) response is chosen as the
    median-score comment among the selected set.
    """

    rng = random.Random(seed)

    posts: Dict[str, Dict] = {}
    for r in records:
        pid = r["post_id"]
        posts.setdefault(pid, {"history": r["history"], "comments": {}})
        for side in ("A", "B"):
            cid = r[f"c_root_id_{side}"]
            txt = r[f"human_ref_{side}"]
            score = float(r[f"score_{side}"])
            if cid not in posts[pid]["comments"]:
                posts[pid]["comments"][cid] = Comment(cid=cid, text=txt, score=score)

    pools: List[PromptPool] = []
    for pid, info in posts.items():
        history = info["history"]
        comments = list(info["comments"].values())
        # Filter empty/very short texts.
        comments = [c for c in comments if c.text and len(c.text.split()) >= 5]
        if len(comments) < k_total:
            continue
        comments.sort(key=lambda c: c.score)

        # Mix extremes: half lowest + half highest.
        half = k_total // 2
        chosen: List[Comment] = comments[:half] + comments[-(k_total - half) :]
        # If duplicates (rare), dedupe and refill.
        uniq = {c.cid: c for c in chosen}
        chosen = list(uniq.values())
        if len(chosen) < k_total:
            remaining = [c for c in comments if c.cid not in uniq]
            rng.shuffle(remaining)
            chosen += remaining[: (k_total - len(chosen))]

        chosen.sort(key=lambda c: c.score)
        baseline = chosen[len(chosen) // 2]
        candidates = [c for c in chosen if c.cid != baseline.cid]
        if len(candidates) < 2:
            continue
        pools.append(PromptPool(post_id=pid, history=history, baseline=baseline, candidates=candidates))

    return pools


def _build_text_index(pools: List[PromptPool]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Flatten all (history, response) pairs into a single list of texts.

    Returns:
      texts: list of concatenated (history + response) strings
      index: list of (post_id, comment_id) keys aligned with texts
    """

    texts: List[str] = []
    index: List[Tuple[str, str]] = []
    for p in pools:
        texts.append(p.history + "\n\n" + p.baseline.text)
        index.append((p.post_id, p.baseline.cid))
        for c in p.candidates:
            texts.append(p.history + "\n\n" + c.text)
            index.append((p.post_id, c.cid))
    return texts, index


def fit_tfidf(
    pools: List[PromptPool], *, max_features: int = 8000
) -> Tuple[TfidfVectorizer, sparse.csr_matrix, List[Tuple[str, str]]]:
    """Fit TF--IDF on concatenated (history + response) strings.

    Returns:
      vectorizer, X (n_items x d), and an index list mapping rows to (post_id,cid).
    """

    texts, index = _build_text_index(pools)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        lowercase=True,
    )
    X = vectorizer.fit_transform(texts).tocsr()
    return vectorizer, X, index


def fit_sbert(
    pools: List[PromptPool],
    *,
    model_name: str,
    batch_size: int = 32,
    cache_path: Optional[str] = None,
) -> Tuple[Callable[[List[str]], np.ndarray], np.ndarray, List[Tuple[str, str]]]:
    """Compute frozen MiniLM/SBERT embeddings (no finetuning).

    We intentionally implement embeddings using ``transformers`` directly
    (mean pooling + L2 normalization) to avoid adding a hard dependency on
    ``sentence-transformers``.

    Args:
      pools: Prompt pools.
      model_name: Hugging Face model id, e.g. ``sentence-transformers/all-MiniLM-L6-v2``.
      batch_size: Encoding batch size.
      cache_path: Optional .npz cache path for embeddings.

    Returns:
      (featurize, X, index) where X is a dense float32 array and
      ``featurize(texts)`` returns MiniLM embeddings for arbitrary strings.
    """

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "SBERT encoder requested, but required packages are missing.\n"
            "Install: pip install transformers torch\n"
            "(torch is often already installed)."
        ) from e

    # Build a featurizer closure that loads the frozen encoder once.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SBERT model '{model_name}'.\n"
            "If you are running offline, pre-download the model (HF cache) or pass a local path."
        ) from e
    model.eval()
    device = "cpu"
    model.to(device)

    def featurize(texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                token_emb = out.last_hidden_state  # (b, L, h)
                attn = enc["attention_mask"].unsqueeze(-1).float()  # (b, L, 1)
                # Mean pooling.
                sent_emb = (token_emb * attn).sum(dim=1) / torch.clamp(attn.sum(dim=1), min=1.0)
                sent_emb = F.normalize(sent_emb, p=2, dim=1)
                all_embs.append(sent_emb.cpu().numpy().astype(np.float32))
        return np.vstack(all_embs)

    texts, index = _build_text_index(pools)

    X: Optional[np.ndarray] = None
    if cache_path is not None and os.path.exists(cache_path):
        try:
            z = np.load(cache_path)
            Xc = z["X"]
            if Xc.shape[0] == len(texts):
                X = Xc
        except Exception:
            X = None

    if X is None:
        X = featurize(texts)
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, X=X)

    return featurize, X, index


def prepare_pools(
    pools: List[PromptPool],
    X: np.ndarray | sparse.csr_matrix,
    index: List[Tuple[str, str]],
) -> List[PreparedPool]:
    """Precompute per-pool sparse diffs X(action)-X(baseline) and score utilities.

    This dramatically speeds up simulation by avoiding per-round sparse stacking.
    """

    row_of: Dict[Tuple[str, str], int] = {k: i for i, k in enumerate(index)}
    prepared: List[PreparedPool] = []
    for p in pools:
        b_row = row_of[(p.post_id, p.baseline.cid)]
        cand_rows = [row_of[(p.post_id, c.cid)] for c in p.candidates]

        X_b = X[b_row]
        X_c = X[cand_rows]
        if sparse.issparse(X_c):
            X_diff = X_c - sparse.vstack([X_b] * X_c.shape[0])
            X_diff = X_diff.tocsr()
        else:
            # Dense: broadcast baseline row.
            X_b_dense = np.asarray(X_b).reshape(1, -1)
            X_diff = np.asarray(X_c) - np.repeat(X_b_dense, repeats=X_c.shape[0], axis=0)

        u_base = math.log(1.0 + p.baseline.score)
        u_cands = np.array([math.log(1.0 + c.score) for c in p.candidates], dtype=float)
        prepared.append(PreparedPool(X_diff=X_diff, u_base=u_base, u_cands=u_cands, u_best=float(u_cands.max())))

    return prepared


def _l2_project_rows(W: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1)
    scale = np.minimum(1.0, radius / (norms + 1e-12))
    return W * scale[:, None]


@dataclass
class AlgoCfg:
    T: int = 250
    n_seeds: int = 3
    N: int = 8
    m: int = 64
    eta0: float = 0.8
    kappa: float = 1.2
    beta: float = 1.0
    prior_sigma: float = 1.0
    proj_radius: float = 5.0
    temp: float = 3.0  # preference noise temperature on log-score differences


def simulate_online(
    pools: List[PreparedPool],
    *,
    cfg: AlgoCfg,
    seed: int,
    variant: str,
    sgld: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one seed. Returns (cum_regret, final_mean_weights)."""

    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Initialize particle ensemble from Gaussian prior.
    d = pools[0].X_diff.shape[1]
    W = rng.normal(scale=cfg.prior_sigma, size=(cfg.N, d))
    W = _l2_project_rows(W, cfg.proj_radius)

    # Replay buffer: store (pool_idx, action_idx, label).
    pool_idxs: List[int] = []
    action_idxs: List[int] = []
    labels: List[float] = []

    regret = np.zeros(cfg.T, dtype=float)

    for t in range(1, cfg.T + 1):
        p_idx = int(rng.integers(0, len(pools)))
        pool = pools[p_idx]
        X_diff = pool.X_diff
        u_base = pool.u_base
        u_cands = pool.u_cands
        u_best = pool.u_best

        # Predictions: logits = <w, diff>
        if sparse.issparse(X_diff):
            logits = X_diff.dot(W.T)  # (K,N)
        else:
            logits = np.asarray(X_diff) @ W.T
        mean = logits.mean(axis=1)
        std = logits.std(axis=1)

        if variant == "ole":
            score = mean + cfg.kappa * std
        elif variant == "mean":
            score = mean
        elif variant == "ts":
            i = int(rng.integers(0, cfg.N))
            score = logits[:, i]
        else:
            raise ValueError(f"unknown variant: {variant}")

        a_idx = int(np.argmax(score))
        chosen_u = float(u_cands[a_idx])
        regret[t - 1] = u_best - chosen_u

        # Noisy preference bit comparing chosen action to baseline.
        p = float(sigmoid(np.array((chosen_u - u_base) / cfg.temp)))
        b = 1.0 if rng.random() < p else 0.0

        pool_idxs.append(p_idx)
        action_idxs.append(a_idx)
        labels.append(b)

        # One (S)GLD update per particle using a mini-batch from the buffer.
        eta = cfg.eta0 / t
        n_data = len(labels)
        mb = min(cfg.m, n_data)
        mb_idx = rng.integers(0, n_data, size=mb)

        # Build a mini-batch of sparse diffs by indexing into prepared pools.
        diffs = [pools[pool_idxs[i]].X_diff[action_idxs[i]] for i in mb_idx]
        if sparse.issparse(diffs[0]):
            D = sparse.vstack(diffs).tocsr()
        else:
            D = np.vstack([np.asarray(v).reshape(1, -1) for v in diffs])
        y = np.array([labels[i] for i in mb_idx], dtype=float)  # (mb,)

        # logits_mb: (mb,N)
        logits_mb = D.dot(W.T) if sparse.issparse(D) else (D @ W.T)
        p_mb = sigmoid(logits_mb)
        coef = (p_mb - y[:, None])  # (mb,N)

        # grad: (N,d). Use (coef^T @ D) where D is sparse.
        if sparse.issparse(D):
            grad = (coef.T @ D).astype(float)
        else:
            grad = (coef.T @ D).astype(float)
        grad *= float(n_data) / float(mb)

        # Gaussian prior.
        grad += cfg.beta * W / (cfg.prior_sigma ** 2)

        if sgld:
            noise = rng.normal(size=W.shape)
            W = W - eta * grad + math.sqrt(2.0 * cfg.beta * eta) * noise
        else:
            W = W - eta * grad

        W = _l2_project_rows(W, cfg.proj_radius)

    return np.cumsum(regret), W.mean(axis=0)


def eval_pairwise_accuracy(
    records: List[dict],
    *,
    featurize,
    w: np.ndarray,
) -> float:
    """Pairwise accuracy on held-out SHP comparisons."""

    correct = 0
    total = 0
    for r in records:
        x = r["history"]
        a = r["human_ref_A"]
        b = r["human_ref_B"]
        y = int(r["labels"])  # 1 if A preferred
        Xa = featurize([x + "\n\n" + a])
        Xb = featurize([x + "\n\n" + b])
        if sparse.issparse(Xa):
            diff = (Xa - Xb)
            logit = float(diff.dot(w))
        else:
            diff = np.asarray(Xa) - np.asarray(Xb)
            logit = float(diff.reshape(-1) @ w)
        pred = 1 if logit >= 0 else 0
        correct += int(pred == y)
        total += 1
    return float(correct) / float(max(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["all", "run"], default="all")
    parser.add_argument("--T", type=int, default=250)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--eta0", type=float, default=0.8)
    parser.add_argument("--kappa", type=float, default=1.2)
    parser.add_argument("--proj_radius", type=float, default=5.0)
    parser.add_argument("--temp", type=float, default=3.0)
    parser.add_argument("--max_features", type=int, default=2000)
    parser.add_argument("--k_total", type=int, default=6)
    parser.add_argument("--encoder", choices=["tfidf", "sbert"], default="tfidf")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sbert_batch_size", type=int, default=32)
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for saved figures/metrics (e.g., 'sbert').",
    )
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    root = _repo_root()
    outdir = args.outdir or os.path.join(root, "experiments", "rlhf_bridge", "results")
    os.makedirs(outdir, exist_ok=True)

    data_dir = os.path.join(root, "experiments", "rlhf_bridge", "data")
    os.makedirs(data_dir, exist_ok=True)
    val_path = os.path.join(data_dir, "shp_askphysics_validation.jsonl")
    test_path = os.path.join(data_dir, "shp_askphysics_test.jsonl")
    download_if_needed(SHP_FILES["askphysics_validation"], val_path)
    download_if_needed(SHP_FILES["askphysics_test"], test_path)

    val = load_jsonl(val_path)
    test = load_jsonl(test_path)

    pools = build_pools(val, k_total=args.k_total, seed=0)
    # The askphysics split is intentionally small; we keep the pool threshold low
    # so the experiment remains lightweight and reproducible.
    if len(pools) < 10:
        raise RuntimeError(f"Too few pools built ({len(pools)}). Try a different split/domain.")

    encoder_meta: Dict[str, object] = {}
    if args.encoder == "tfidf":
        vectorizer, X, index = fit_tfidf(pools, max_features=args.max_features)
        featurize = vectorizer.transform
        encoder_meta = {"max_features": args.max_features, "ngram_range": "(1,2)", "min_df": 2}
    else:
        cache_path = os.path.join(data_dir, f"sbert_cache_{args.tag or 'sbert'}.npz")
        featurize, X, index = fit_sbert(
            pools,
            model_name=args.sbert_model,
            batch_size=args.sbert_batch_size,
            cache_path=cache_path,
        )
        encoder_meta = {"model": args.sbert_model, "batch_size": args.sbert_batch_size, "cache_path": cache_path}
    prepared = prepare_pools(pools, X, index)

    cfg = AlgoCfg(
        T=args.T,
        n_seeds=args.n_seeds,
        N=args.N,
        m=args.m,
        eta0=args.eta0,
        kappa=args.kappa,
        proj_radius=args.proj_radius,
        temp=args.temp,
    )

    variants = [
        ("ole", True, "OLE (UCB + SGLD)"),
        ("mean", True, "Posterior mean (SGLD)"),
        ("ts", True, "Thompson (SGLD)"),
    ]

    curves: Dict[str, List[np.ndarray]] = {name: [] for _, __, name in variants}
    final_ws: Dict[str, List[np.ndarray]] = {name: [] for _, __, name in variants}

    for s in range(cfg.n_seeds):
        base_seed = 1234 + s
        for variant, sgld, name in variants:
            curve, w = simulate_online(prepared, cfg=cfg, seed=base_seed, variant=variant, sgld=sgld)
            curves[name].append(curve)
            final_ws[name].append(w)

    # Aggregate curves.
    mean_curves = {}
    se_curves = {}
    for name, runs in curves.items():
        A = np.stack(runs, axis=0)
        mean_curves[name] = A.mean(axis=0)
        ddof = 1 if A.shape[0] > 1 else 0
        se_curves[name] = A.std(axis=0, ddof=ddof) / math.sqrt(A.shape[0])

    # Held-out pairwise accuracy using the *mean* of final weights across seeds.
    acc = {}
    for name, ws in final_ws.items():
        w_bar = np.mean(np.stack(ws, axis=0), axis=0)
        acc[name] = eval_pairwise_accuracy(test, featurize=featurize, w=w_bar)

    # Plot.
    suffix = f"_{args.tag}" if args.tag else ""
    fig_path = os.path.join(root, "figures", f"shp_bridge_regret{suffix}.pdf")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure(figsize=(6.0, 3.2))
    t = np.arange(1, cfg.T + 1)
    for name in mean_curves:
        plt.plot(t, mean_curves[name], label=name)
        plt.fill_between(t, mean_curves[name] - se_curves[name], mean_curves[name] + se_curves[name], alpha=0.2)
    plt.xlabel("round t")
    plt.ylabel("cumulative regret (log-score units)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    # Regret gap plot: improvement over posterior-mean baseline.
    mean_name = "Posterior mean (SGLD)"
    gap_fig_path = os.path.join(root, "figures", f"shp_bridge_regret_gap{suffix}.pdf")
    plt.figure(figsize=(6.0, 3.2))
    t = np.arange(1, cfg.T + 1)
    for name, runs in curves.items():
        if name == mean_name:
            continue
        # Per-seed gap curves to get an accurate SE.
        gaps = []
        for s in range(cfg.n_seeds):
            gaps.append(curves[mean_name][s] - curves[name][s])
        G = np.stack(gaps, axis=0)
        m_gap = G.mean(axis=0)
        ddof = 1 if G.shape[0] > 1 else 0
        se_gap = G.std(axis=0, ddof=ddof) / math.sqrt(G.shape[0])
        plt.plot(t, m_gap, label=name)
        plt.fill_between(t, m_gap - se_gap, m_gap + se_gap, alpha=0.2)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("round t")
    plt.ylabel("regret reduction vs. posterior mean")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(gap_fig_path)
    plt.close()

    metrics = {
        "cfg": cfg.__dict__,
        "encoder": args.encoder,
        "encoder_meta": encoder_meta,
        "tag": args.tag,
        "num_pools": len(pools),
        "heldout_pairwise_accuracy": acc,
        "final_cum_regret": {name: float(mean_curves[name][-1]) for name in mean_curves},
    }
    metrics_path = os.path.join(outdir, f"shp_bridge_metrics{suffix}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    # Also write a small LaTeX snippet with macros used by the paper.
    metrics_tex_path = os.path.join(outdir, f"shp_bridge_metrics{suffix}.tex")
    with open(metrics_tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiments/rlhf_bridge/run_shp_bridge.py.\n")
        f.write(f"\\newcommand{{\\shpBridgeT}}{{{cfg.T}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeN}}{{{cfg.N}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeNumSeeds}}{{{cfg.n_seeds}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeNumPools}}{{{len(pools)}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeKappa}}{{{cfg.kappa}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeRegretOLE}}{{{metrics['final_cum_regret']['OLE (UCB + SGLD)']:.1f}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeRegretMean}}{{{metrics['final_cum_regret']['Posterior mean (SGLD)']:.1f}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeRegretTS}}{{{metrics['final_cum_regret']['Thompson (SGLD)']:.1f}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeAccOLE}}{{{metrics['heldout_pairwise_accuracy']['OLE (UCB + SGLD)']:.3f}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeAccMean}}{{{metrics['heldout_pairwise_accuracy']['Posterior mean (SGLD)']:.3f}}}\n")
        f.write(f"\\newcommand{{\\shpBridgeAccTS}}{{{metrics['heldout_pairwise_accuracy']['Thompson (SGLD)']:.3f}}}\n")
    print("Saved:")
    print(" -", fig_path)
    print(" -", gap_fig_path)
    print(" -", metrics_path)
    print(" -", metrics_tex_path)


if __name__ == "__main__":
    main()
