#!/usr/bin/env python3
import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import tqdm
import torch
from tqdm.contrib.concurrent import process_map
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from math_dapo import compute_score


def jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    return x


# ---------- compute_score parallel ----------
def verify_one(arg):
    rsp, ground_truth = arg
    return compute_score(rsp, ground_truth)


# ---------- RM single GPU (NO device_map) ----------
def load_reward_model_single_gpu(
    rm_path: str,
    device: str = "cuda:0",
    use_flash_attn: bool = True,
):
    """
    IMPORTANT:
    - Do NOT use device_map=... here, to avoid transformers warmup calling mem_get_info
      with a bad ordinal under CUDA_VISIBLE_DEVICES.
    - Instead: load on CPU then .to(device).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but RM scoring expects CUDA.")

    # If CUDA_VISIBLE_DEVICES is set to a single GPU, device should be "cuda:0"
    if device != "cuda:0":
        # still allow user to pass cuda:0 only; keep strict to avoid ordinal mismatch
        raise ValueError(f"For single-GPU RM, please use --rm_device cuda:0 (got {device})")

    # Ensure current device is 0 in this visible namespace
    torch.cuda.set_device(0)

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        num_labels=1,
    )
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    # Load on CPU first (default), then move to GPU
    rm = AutoModelForSequenceClassification.from_pretrained(rm_path, **kwargs)
    tok = AutoTokenizer.from_pretrained(rm_path)

    rm.to(device)
    rm.eval()
    return rm, tok, device


def build_conv(prompt_messages: List[Dict[str, str]], response_text: str) -> List[Dict[str, str]]:
    conv = []
    for m in prompt_messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            conv.append({"role": m["role"], "content": m["content"]})
    conv.append({"role": "assistant", "content": response_text})
    return conv


@torch.inference_mode()
def orm_score_batch(
    rm,
    rm_tokenizer,
    device: str,
    prompts: List[List[Dict[str, str]]],
    responses: List[str],
    max_length: int,
) -> List[float]:
    assert len(prompts) == len(responses)

    texts = []
    for p, r in zip(prompts, responses):
        conv = build_conv(p, r)
        s = rm_tokenizer.apply_chat_template(conv, tokenize=False)
        texts.append(s)

    enc = rm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    ).to(device)

    logits = rm(**enc).logits
    scores = logits.squeeze(-1).float().detach().cpu().tolist()
    return [float(x) for x in scores]


def main(args):
    os.makedirs(os.path.dirname(args.detail_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)

    with open(args.gen_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("items", [])
    if not items:
        raise ValueError(f"No items found in {args.gen_json}")

    # ---------- flatten for compute_score ----------
    flat_pairs_score: List[Tuple[str, str]] = []
    index_map: List[Tuple[int, int]] = []  # (i, j)

    for i, it in enumerate(items):
        gt = str(it["ground_truth"])
        rsps = it["responses"]
        for j, rsp in enumerate(rsps):
            flat_pairs_score.append((rsp, gt))
            index_map.append((i, j))

    total_pairs = len(flat_pairs_score)

    # ---------- 1) compute_score ----------
    flat_res = process_map(
        verify_one,
        flat_pairs_score,
        max_workers=args.verify_workers,
        chunksize=1,
    )

    # attach placeholders
    for it in items:
        it["answers"] = [None] * len(it["responses"])

    # fill compute_score results
    for (i, j), r in zip(index_map, flat_res):
        items[i]["answers"][j] = {
            "response_raw": items[i]["responses"][j],
            "pred": r.get("pred", None),
            "acc": float(r.get("acc", 0.0)),
            "score": r.get("score", None),
            "orm_score": None,  # to fill
        }

    # ---------- 2) ORM scoring (single GPU, batched) ----------
    # strongly recommend: CUDA_VISIBLE_DEVICES set to ONE GPU, and rm_device=cuda:0
    rm, rm_tok, rm_dev = load_reward_model_single_gpu(
        args.rm_path, device=args.rm_device, use_flash_attn=args.rm_flash_attn
    )

    # Prepare flatten prompts/responses in same order as index_map
    flat_prompts = []
    flat_responses = []
    for it in items:
        prompt = it["prompt"]
        for rsp in it["responses"]:
            flat_prompts.append(prompt)
            flat_responses.append(rsp)

    orm_scores: List[float] = []
    with tqdm.tqdm(total=total_pairs, desc=f"ORM scoring (batched, {rm_dev})") as pbar:
        for s in range(0, total_pairs, args.rm_batch_size):
            e = min(s + args.rm_batch_size, total_pairs)
            scores = orm_score_batch(
                rm=rm,
                rm_tokenizer=rm_tok,
                device=rm_dev,
                prompts=flat_prompts[s:e],
                responses=flat_responses[s:e],
                max_length=args.rm_max_length,
            )
            orm_scores.extend(scores)
            pbar.update(e - s)

    assert len(orm_scores) == total_pairs

    # write back orm scores
    for (i, j), orm in zip(index_map, orm_scores):
        items[i]["answers"][j]["orm_score"] = float(orm)

    # ---------- 3) Metrics & aggregation ----------
    detail_records = []
    agg: Dict[str, Dict[str, float]] = {}

    for it in items:
        ds = str(it["data_source"])
        answers = it["answers"]
        accs = [a["acc"] for a in answers]
        orms = [a["orm_score"] for a in answers]

        K = len(answers)
        pass1 = float(accs[0]) if K else 0.0
        passk = float(1.0 if any(a >= 1.0 for a in accs) else 0.0)
        meank = float(sum(accs) / K) if K else 0.0

        orm_k = float(sum(orms) / K) if K else 0.0
        best_idx = int(np.argmax(orms)) if K else 0
        best_orm = float(max(orms)) if K else 0.0  # CHANGED: best@K is best_orm

        sol_len = float(len(answers[0]["response_raw"])) if K else 0.0

        record = {
            "question": it["question"],
            "ground_truth": it["ground_truth"],
            "data_source": ds,
            "answers": answers,
            "metrics": {
                "pass@1": pass1,
                "orm@K": orm_k,
                "best@K": best_orm,   # CHANGED
                "pass@K": passk,
                "mean@K": meank,
                "best_idx": best_idx,
                "K": K,
            },
        }
        detail_records.append(record)

        if ds not in agg:
            agg[ds] = {
                "count": 0.0,
                "sum_pass1": 0.0,
                "sum_ormk": 0.0,
                "sum_bestk": 0.0,
                "sum_passk": 0.0,
                "sum_meank": 0.0,
                "sum_len": 0.0,
            }
        agg[ds]["count"] += 1.0
        agg[ds]["sum_pass1"] += pass1
        agg[ds]["sum_ormk"] += orm_k
        agg[ds]["sum_bestk"] += best_orm
        agg[ds]["sum_passk"] += passk
        agg[ds]["sum_meank"] += meank
        agg[ds]["sum_len"] += sol_len

    summary_records = []
    for ds, a in agg.items():
        n = int(a["count"])
        summary_records.append({
            "data_source": ds,
            "count": n,
            "avg_pass@1": a["sum_pass1"] / n,
            "avg_orm@K": a["sum_ormk"] / n,
            "avg_best@K": a["sum_bestk"] / n,
            "avg_pass@K": a["sum_passk"] / n,
            "avg_mean@K": a["sum_meank"] / n,
            "avg_solution_str_len": a["sum_len"] / n,
        })

    with open(args.detail_json, "w", encoding="utf-8") as f:
        json.dump(jsonable({"meta": payload.get("meta", {}), "detail": detail_records}),
                  f, ensure_ascii=False, indent=2)

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(jsonable({"meta": payload.get("meta", {}), "summary": sorted(summary_records, key=lambda x: x["data_source"])}),
                  f, ensure_ascii=False, indent=2)

    print(f"[DONE] detail:  {args.detail_json}")
    print(f"[DONE] summary: {args.summary_json}")

    print("\n[SUMMARY]")
    for r in sorted(summary_records, key=lambda x: x["data_source"]):
        print(
            f"- {r['data_source']}: n={r['count']}, "
            f"Pass@1={r['avg_pass@1']:.4f}, ORM@K={r['avg_orm@K']:.4f}, Best@K={r['avg_best@K']:.4f}, "
            f"Pass@K={r['avg_pass@K']:.4f}, Mean@K={r['avg_mean@K']:.4f}, "
            f"Len={r['avg_solution_str_len']:.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gen_json", type=str, required=True)
    parser.add_argument("--verify_workers", type=int, default=50)

    parser.add_argument("--rm_path", type=str, required=True)

    # single GPU only; must be cuda:0 when CUDA_VISIBLE_DEVICES is single GPU
    parser.add_argument("--rm_device", type=str, default="cuda:0")

    # Bigger default batch size; tune if OOM
    parser.add_argument("--rm_batch_size", type=int, default=128)
    parser.add_argument("--rm_max_length", type=int, default=8192)

    parser.add_argument("--rm_flash_attn", action="store_true")
    parser.set_defaults(rm_flash_attn=True)

    parser.add_argument("--detail_json", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)

    args = parser.parse_args()
    print(args)
    main(args)
