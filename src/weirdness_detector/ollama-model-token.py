#!/usr/bin/env python3
"""
Validate ontology-generated sentences using HuggingFace token log-likelihood scoring.

Reads an input text file (one sentence per line), scores each sentence using a
local HuggingFace causal LM (teacher-forced prompt token logprobs), classifies
coherence into {-1, 0, 1}, and writes results to a CSV file with fields:

1) sentence
2) label   (-1 definitely incoherent, 0 possibly coherent, 1 definitely coherent)
3) avg_nll
4) p95
5) run

Usage:
  python ollama-model-token.py --model meta-llama/Meta-Llama-3-8B --input in.txt --output out.csv
"""

import argparse
import csv
import math
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


class HFSentenceValidator:
    def __init__(self, model: str, device: str = "cpu", dtype: str = "auto"):
        self.model = model
        self.device = device
        self.dtype = _parse_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.model_obj = AutoModelForCausalLM.from_pretrained(model, torch_dtype=self.dtype)
        self.model_obj.to(self.device)
        self.model_obj.eval()

    def score(self, sentence: str, return_tokens: bool = False) -> Dict[str, float]:
        """
        Returns metrics:
          - avg_nll: average negative log-likelihood per token (nats)
          - p95: 95th percentile surprisal (nats)
          - run: longest run of surprisals above a threshold
        """
        encoded = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]
        if (
            self.tokenizer.eos_token_id is not None
            and input_ids[0, -1].item() == self.tokenizer.eos_token_id
        ):
            input_ids = input_ids[:, :-1]
        if input_ids.shape[1] < 2:
            raise ValueError("Sentence is too short to score.")
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model_obj(input_ids=input_ids)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, 1:]
            token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logprobs_list = token_logprobs[0].tolist()
        token_texts = None
        if return_tokens:
            token_texts = self.tokenizer.convert_ids_to_tokens(target_ids[0].tolist())

        # token_logprobs are log(P(token | previous_tokens)) in natural log (nats), typically <= 0.
        # surprisal = -logprob (>= 0)
        surprisals: List[float] = []
        for lp in token_logprobs_list:
            surprisals.append(-float(lp))

        n = len(surprisals)
        avg_nll = sum(surprisals) / n

        # 95th percentile surprisal (robust "bad typical token" measure)
        p95 = self._percentile(surprisals, 0.95)

        # Longest run of "low-likelihood" tokens: surprisal above threshold
        run = float(self._longest_run(surprisals, threshold=3.5))  # keep numeric for CSV consistency

        result = {"avg_nll": float(avg_nll), "p95": float(p95), "run": run}
        if return_tokens:
            result["tokens"] = token_texts
            result["token_logprobs"] = token_logprobs_list
        return result

    @staticmethod
    def _longest_run(values: List[float], threshold: float) -> int:
        longest = 0
        current = 0
        for v in values:
            if v > threshold:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 0
        return longest

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        """
        Deterministic percentile using "nearest-rank"-style indexing on sorted values.
        q in [0,1]. For q=0.95, returns an element such that ~95% of values are <= it.
        """
        if not values:
            raise ValueError("Cannot compute percentile of empty list.")
        if q <= 0:
            return float(min(values))
        if q >= 1:
            return float(max(values))

        vs = sorted(values)
        # index in [0, len-1]
        idx = int(math.floor(q * (len(vs) - 1)))
        return float(vs[idx])


def classify(avg_nll: float, p95: float, run: float) -> int:
    """
    3-way coherence classification:
      1  => definitely coherent
      0  => possibly coherent
     -1  => definitely incoherent

    These thresholds are reasonable starting points for LLaMA-class models.
    You should calibrate per model and domain if you want maximum accuracy.
    """
    # Definitely incoherent: any strong failure signal
    if avg_nll > 3.5 or p95 > 7.0 or run >= 4:
        return -1

    # Definitely coherent: strong fluency with minimal local pathologies
    if avg_nll < 2.2 and p95 < 5.0 and run <= 1:
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score and classify sentence coherence using HuggingFace; output CSV."
    )
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--input", required=True, help="Input text file (one sentence per line)")
    p.add_argument("--output", required=True, help="Output CSV file")
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu, cuda, cuda:0)",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype: auto, float16, bfloat16, float32",
    )
    p.add_argument(
        "--print-token-scores",
        action="store_true",
        help="Print per-token logprobs and surprisals to stdout",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    validator = HFSentenceValidator(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
    )

    # Use newline="" for correct CSV writing on Windows too.
    with open(args.input, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    total_lines = len(lines)
    print(f"Total lines in input: {total_lines}")

    with open(args.output, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["sentence", "label", "avg_nll", "p95", "run"])

        processed = 0
        for line_num, raw in enumerate(lines, start=1):
            print(f"Progress: line {line_num}/{total_lines}")
            sentence = raw.strip()
            if not sentence:
                continue

            try:
                processed += 1
                print(f"Scoring line {line_num} (processed {processed})")
                metrics = validator.score(sentence, return_tokens=args.print_token_scores)
                avg_nll = metrics["avg_nll"]
                p95 = metrics["p95"]
                run = metrics["run"]
                label = classify(avg_nll, p95, run)

                writer.writerow([sentence, label, f"{avg_nll:.6f}", f"{p95:.6f}", f"{run:.0f}"])
                if args.print_token_scores:
                    tokens = metrics.get("tokens") or []
                    token_logprobs = metrics.get("token_logprobs") or []
                    print(f"Token scores for line {line_num}:")
                    for idx, (token, lp) in enumerate(zip(tokens, token_logprobs), start=1):
                        print(f"  {idx:03d} token={token!r} logprob={lp:.6f} surprisal={-lp:.6f}")

            except Exception:
                print(f"Ollama scoring failed on line {line_num}: {sentence}")
                raise


if __name__ == "__main__":
    main()
