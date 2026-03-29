#!/usr/bin/env python3
import os
import json
import time
import asyncio
import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tqdm
from openai import AsyncOpenAI


def build_client(base_url: str, api_key: str, timeout: int) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def to_list_prompt(p: Any) -> List[Dict[str, str]]:
    if isinstance(p, np.ndarray):
        p = p.tolist()
    if isinstance(p, list):
        return p
    if isinstance(p, dict):
        return [p]
    if isinstance(p, str):
        return [{"role": "user", "content": p}]
    return [{"role": "user", "content": "" if p is None else str(p)}]


def extract_question_text(messages: List[Dict[str, str]]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


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


async def chat(client: AsyncOpenAI, messages, temperature, top_p, max_tokens, model):
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    return completion


async def run_in_chunks(tasks: List, chunk_size: int):
    out = []
    start = time.time()
    with tqdm.tqdm(total=len(tasks), desc="LLM generations") as pbar:
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i : i + chunk_size]
            rets = await asyncio.gather(*chunk)
            out.extend(rets)
            pbar.update(len(chunk))
    end = time.time()
    print(f"[INFO] Time taken: {end - start:.2f}s for {len(tasks)} requests")
    return out


def main(args):
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    if os.path.exists(args.out_json) and not args.force:
        print(f"[INFO] {args.out_json} exists; skip (use --force to overwrite).")
        return

    df = pd.read_parquet(args.test_file, engine="pyarrow")
    for col in ["prompt", "reward_model", "data_source"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.test_file}. Columns={list(df.columns)}")

    if args.num is not None and args.num > 0:
        df = df.head(args.num).copy()

    prompts = [to_list_prompt(p) for p in df["prompt"].tolist()]
    questions = [extract_question_text(p) for p in prompts]
    data_sources = df["data_source"].astype(str).tolist()
    reward_models = df["reward_model"].tolist()

    ground_truths = []
    for rm in reward_models:
        if not isinstance(rm, dict) or "ground_truth" not in rm:
            raise ValueError("reward_model must be dict with key 'ground_truth'")
        ground_truths.append(str(rm["ground_truth"]))

    client = build_client(args.base_url, args.api_key, args.timeout)

    tasks = []
    for i in range(len(df)):
        for _ in range(args.n_samples):
            tasks.append(
                chat(
                    client,
                    prompts[i],
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.model,  # IMPORTANT: LoRA name
                )
            )

    rets = asyncio.run(run_in_chunks(tasks, chunk_size=args.concurrency))
    all_outputs = [r.choices[0].message.content for r in rets]

    outputs_2d: List[List[str]] = []
    k = 0
    for i in range(len(df)):
        row = []
        for _ in range(args.n_samples):
            row.append(all_outputs[k])
            k += 1
        outputs_2d.append(row)

    payload = {
        "meta": {
            "test_file": args.test_file,
            "num": args.num,
            "model": args.model,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "base_url": args.base_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
        "items": [],
    }

    for i in range(len(df)):
        payload["items"].append({
            "idx": i,
            "data_source": data_sources[i],
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "prompt": jsonable(prompts[i]),
            "reward_model": jsonable(reward_models[i]),
            "responses": outputs_2d[i],  # list[str], length = n_samples
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote generations: {args.out_json} | items={len(payload['items'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="NOT A REAL KEY")
    parser.add_argument("--timeout", type=int, default=36000)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=20480)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=256)

    parser.add_argument("--model", type=str, required=True)      # LoRA name
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    print(args)
    main(args)
