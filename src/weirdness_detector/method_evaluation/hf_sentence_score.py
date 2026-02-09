#!/usr/bin/env python3
"""
hf_sentence_score.py

Score synthetic sentences using a HuggingFace causal LM (teacher-forced token logprobs)
and compute pathology-oriented features to help filter incoherent sentences.

Input:  text file (one sentence per line)
Output: CSV with columns (no label column; you will join human labels separately)

Default output columns:
  sentence
  n_tokens
  avg_nll
  p95
  p99
  max_surp
  spike_frac_35
  spike_frac_50
  run_norm_35
  uniq_token_ratio
  rep_1gram
  rep_2gram
  top1_prob_mean
  top1_minus_true_mean
  punct_ratio
  digit_ratio
  upper_ratio
  paren_imbalance_abs
  bracket_imbalance_abs

Notes:
- surprisals are in nats (natural log).
- spike_frac_* measure fraction of tokens with surprisal above a threshold.
- run_norm_35 = longest run of surprisal>3.5 divided by n_tokens (reduces length bias).
- rep_1gram, rep_2gram detect degeneracy/repetition.
- top1_prob_mean and top1_minus_true_mean capture "model confusion":
    - top1_prob_mean: average probability of model's top predicted token at each position
    - top1_minus_true_mean: average (top1_prob - true_token_prob) at each position

Usage:
  python hf_sentence_score.py --model meta-llama/Meta-Llama-3-8B --input in.txt --output scored.csv --device cuda --dtype float16
"""

import argparse
import csv
import math
import string
from typing import Dict, List, Optional, Tuple

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


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


class HFSentenceScorer:
    def __init__(self, model: str, device: str = "cpu", dtype: str = "auto"):
        self.model = model
        self.device = device
        self.dtype = _parse_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.model_obj = AutoModelForCausalLM.from_pretrained(model, torch_dtype=self.dtype)
        self.model_obj.to(self.device)
        self.model_obj.eval()

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            raise ValueError("Cannot compute percentile of empty list.")
        if q <= 0:
            return float(min(values))
        if q >= 1:
            return float(max(values))
        vs = sorted(values)
        idx = int(math.floor(q * (len(vs) - 1)))
        return float(vs[idx])

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
    def _repetition_features(token_ids: List[int]) -> Tuple[float, float, float]:
        """
        Returns:
          uniq_token_ratio: unique tokens / n
          rep_1gram: fraction of positions where token[i] == token[i-1]
          rep_2gram: fraction of bigrams that repeat (approx)
        """
        n = len(token_ids)
        if n <= 1:
            return 1.0, 0.0, 0.0

        uniq_token_ratio = len(set(token_ids)) / n

        rep_1 = sum(1 for i in range(1, n) if token_ids[i] == token_ids[i - 1])
        rep_1gram = rep_1 / (n - 1)

        # bigram repetition: count repeated bigram occurrences / total bigrams
        bigrams = [(token_ids[i - 1], token_ids[i]) for i in range(1, n)]
        if not bigrams:
            rep_2gram = 0.0
        else:
            seen = {}
            repeats = 0
            for bg in bigrams:
                seen[bg] = seen.get(bg, 0) + 1
            for cnt in seen.values():
                if cnt > 1:
                    repeats += cnt
            rep_2gram = repeats / len(bigrams)

        return float(uniq_token_ratio), float(rep_1gram), float(rep_2gram)

    @staticmethod
    def _text_shape_features(sentence: str) -> Dict[str, float]:
        """
        Cheap, helpful signals for malformed/symbol-soup sentences.
        """
        s = sentence
        n = len(s)
        if n == 0:
            return {
                "punct_ratio": 0.0,
                "digit_ratio": 0.0,
                "upper_ratio": 0.0,
                "paren_imbalance_abs": 0.0,
                "bracket_imbalance_abs": 0.0,
            }

        punct_set = set(string.punctuation)
        punct = sum(1 for ch in s if ch in punct_set)
        digits = sum(1 for ch in s if ch.isdigit())
        uppers = sum(1 for ch in s if ch.isupper())

        paren_imb = abs(s.count("(") - s.count(")"))
        bracket_imb = abs(s.count("[") - s.count("]")) + abs(s.count("{") - s.count("}"))

        return {
            "punct_ratio": punct / n,
            "digit_ratio": digits / n,
            "upper_ratio": uppers / n,
            "paren_imbalance_abs": float(paren_imb),
            "bracket_imbalance_abs": float(bracket_imb),
        }

    def score(self, sentence: str) -> Dict[str, float]:
        """
        Computes:
          - n_tokens (target tokens)
          - avg_nll
          - p95, p99, max_surp
          - spike_frac_35, spike_frac_50
          - run_norm_35
          - uniq_token_ratio, rep_1gram, rep_2gram
          - top1_prob_mean, top1_minus_true_mean
          - text-shape features
        """
        encoded = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]

        # Strip trailing EOS if tokenizer added it
        if (
            self.tokenizer.eos_token_id is not None
            and input_ids.shape[1] >= 1
            and input_ids[0, -1].item() == self.tokenizer.eos_token_id
        ):
            input_ids = input_ids[:, :-1]

        if input_ids.shape[1] < 2:
            raise ValueError("Sentence is too short to score.")

        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model_obj(input_ids=input_ids)
            logits = outputs.logits  # [1, seq, vocab]
            # shift: predict token t given previous tokens
            logits = logits[:, :-1, :]         # [1, seq-1, vocab]
            target_ids = input_ids[:, 1:]      # [1, seq-1]

            log_probs = torch.log_softmax(logits, dim=-1)  # [1, seq-1, vocab]
            token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, seq-1]

            # top-1 prob each position
            top1_logprobs, _ = torch.max(log_probs, dim=-1)  # [1, seq-1]
            top1_probs = torch.exp(top1_logprobs)
            true_probs = torch.exp(token_logprobs)

        token_logprobs_list = token_logprobs[0].tolist()
        target_id_list = target_ids[0].tolist()

        # surprisal = -logprob
        surprisals = [-float(lp) for lp in token_logprobs_list]
        n_tokens = len(surprisals)

        avg_nll = sum(surprisals) / n_tokens
        p95 = self._percentile(surprisals, 0.95)
        p99 = self._percentile(surprisals, 0.99)
        max_surp = float(max(surprisals))

        spike_frac_35 = _safe_div(sum(1 for s in surprisals if s > 3.5), n_tokens)
        spike_frac_50 = _safe_div(sum(1 for s in surprisals if s > 5.0), n_tokens)

        run_35 = self._longest_run(surprisals, threshold=3.5)
        run_norm_35 = _safe_div(float(run_35), float(n_tokens))

        uniq_token_ratio, rep_1gram, rep_2gram = self._repetition_features(target_id_list)

        top1_prob_mean = float(torch.mean(top1_probs).item())
        top1_minus_true_mean = float(torch.mean(top1_probs - true_probs).item())

        shape = self._text_shape_features(sentence)

        return {
            "n_tokens": float(n_tokens),
            "avg_nll": float(avg_nll),
            "p95": float(p95),
            "p99": float(p99),
            "max_surp": float(max_surp),
            "spike_frac_35": float(spike_frac_35),
            "spike_frac_50": float(spike_frac_50),
            "run_norm_35": float(run_norm_35),
            "uniq_token_ratio": float(uniq_token_ratio),
            "rep_1gram": float(rep_1gram),
            "rep_2gram": float(rep_2gram),
            "top1_prob_mean": float(top1_prob_mean),
            "top1_minus_true_mean": float(top1_minus_true_mean),
            **shape,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score sentences with HF LM and write pathology features to CSV.")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--input", required=True, help="Input text file (one sentence per line)")
    p.add_argument("--output", required=True, help="Output CSV file")
    p.add_argument("--device", default="cpu", help="Device: cpu, cuda, cuda:0, etc.")
    p.add_argument("--dtype", default="auto", help="Model dtype: auto, float16, bfloat16, float32")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scorer = HFSentenceScorer(model=args.model, device=args.device, dtype=args.dtype)

    with open(args.input, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    total_lines = len(lines)
    print(f"Total lines in input: {total_lines}")

    # Define output columns explicitly (stable schema)
    fieldnames = [
        "sentence",
        "n_tokens",
        "avg_nll",
        "p95",
        "p99",
        "max_surp",
        "spike_frac_35",
        "spike_frac_50",
        "run_norm_35",
        "uniq_token_ratio",
        "rep_1gram",
        "rep_2gram",
        "top1_prob_mean",
        "top1_minus_true_mean",
        "punct_ratio",
        "digit_ratio",
        "upper_ratio",
        "paren_imbalance_abs",
        "bracket_imbalance_abs",
    ]

    with open(args.output, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0
        for line_num, raw in enumerate(lines, start=1):
            sentence = raw.strip()
            if not sentence:
                continue
            processed += 1
            if processed % 50 == 0 or processed == 1:
                print(f"Scoring {processed}/{total_lines} (line {line_num})")

            try:
                metrics = scorer.score(sentence)
                row = {"sentence": sentence}
                row.update(metrics)

                # Ensure all fields present
                for k in fieldnames:
                    if k not in row:
                        row[k] = ""

                # Format floats consistently
                for k, v in row.items():
                    if isinstance(v, float):
                        row[k] = f"{v:.6f}"

                writer.writerow(row)

            except Exception as e:
                print(f"Scoring failed on line {line_num}: {sentence}")
                raise e

    print(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
