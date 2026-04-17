import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import requests
from tqdm import tqdm

# Set up Ollama endpoint
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Adjust based on your setup
# MODEL_NAME = "custom-model-2-single"  # Replace with your specific model
# MODEL_NAME = "custom-model-2-single:latest"  # Replace with your specific model
MODEL_NAME = "llama3.2:latest"  # Replace with your specific model
# MODEL_NAME = "llama3.2:3b-instruct-fp16"  # Replace with your specific model
# MODEL_NAME = "llama3.3"  # Replace with your specific model
# MODEL_NAME = "deepseek-r1:7b"  # Replace with your specific model
# MODEL_NAME = "gpt-oss:latest"


def evaluate_sentence(sentence, mode):
    time.sleep(0.1)
    prompt = ""
    if mode == "STRICT":
        #  Formulate the prompt (STRICT PROMPT) ~4-5% Sentences classified as Valid
        prompt = f"""
            Evaluate the following sentence for coherence and plausibility:

            Sentence: '{sentence}'

            Classify the sentence as 'Valid' if it makes sense, can logically appear in a book or newspaper, and is applicable to everyday tasks. Focus primarily on whether the object can logically be used with the given verb in a typical everyday situation without overcomplicating the analysis.

            Classify the sentence as 'Invalid' if it is illogical, self-contradictory, or impossible within commonly understood contexts.

            Return just one word 'Valid' or 'Invalid' with a brief explanation about your decision!
        """
    elif mode == "RELAXED":
        # Relaxed Prompt
        prompt = f"""

            Classify a sentence as Invalid only if it describes a scenario that is completely beyond any conceivable reality, even under the most imaginative or hypothetical conditions. This includes cases where:

            The action described is fundamentally impossible under any realistic or fictional context.
            The sentence contains elements that contradict basic universal concepts (e.g., logical impossibilities, contradictions with common human experience).
            In all other cases, classify the sentence as Valid, allowing for unusual, rare, or imaginative scenarios that could happen under specific or extraordinary circumstances. If a human can conceive of the event happening in some form—no matter how unlikely—it should be considered valid.

            Examples:

            The monk borrowed a lion from Cheyenne. → Valid, since borrowing exotic animals, though rare, is possible.

            Victoria was cleaning a locomotive. → Valid, since this is a common, realistic task.

            Duncan doesn't say that "The anthropologist won't be smoking a motion picture." → Valid, because people can express anything, even nonsense.

            Constance will be dreaming of a paper. → Valid, since dreams can contain anything imaginable.

            Gary farms a human corpse. → Valid, since farming techniques could be metaphorically or ethically debated but not physically impossible.

            The refugee will suffer a vehicle brake. → Valid, as metaphorical interpretations could apply in an abstract sense.

            The mountain danced with joy. → Invalid, as inanimate objects do not possess emotions or mobility in any conceivable context.

            Time traveled back into itself to rewrite history. → Invalid, as it contradicts fundamental concepts of causality.

            Return just one word 'Valid' or 'Invalid' nothing else!

            Sentence: '{sentence}'
        """
    else:
        print("INVALID MODE!")

    # Send the request to Ollama
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 5,
            "stream": True,
            "options": {
                "temperature": 0
            }
        },
        stream=True
    )

    if response.status_code == 200:
        full_response = ""
        try:
            # Process each line of the streaming response
            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line)
                    # Append the response text
                    full_response += line_data.get("response", "")
                    # Break if the response is marked as complete
                    if line_data.get("done", False):
                        break
            return full_response
        except json.JSONDecodeError:
            return "Error: Unable to parse JSON in streaming response."
    else:
        return f"Error: {response.status_code} - {response.text}"


def classify_result(result):
    """Returns 1 (valid), -1 (invalid), or None (unclassified/error)."""
    if "Invalid" in result:
        return -1
    if "Valid" in result:
        return 1
    return None


class IncrementalCsvWriter:
    """Thread-safe, incrementally-flushing CSV writer. Columns: line_num, sentence, label."""

    FIELDNAMES = ["line_num", "sentence", "label"]

    def __init__(self, path, append=False):
        self._lock = threading.Lock()
        mode = "a" if append else "w"
        self._file = open(path, mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if not append:
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, line_num, sentence, label):
        """Write one row and flush immediately. Thread-safe."""
        with self._lock:
            self._writer.writerow({"line_num": line_num, "sentence": sentence, "label": label})
            self._file.flush()

    def close(self):
        with self._lock:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_checkpoint(output_path):
    """Read already-processed line numbers from an existing CSV checkpoint.
    Returns a set of ints representing line_num values already written to disk."""
    done = set()
    if not output_path.exists():
        return done
    try:
        with output_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "line_num" in reader.fieldnames:
                for row in reader:
                    try:
                        done.add(int(row["line_num"]))
                    except (ValueError, KeyError):
                        pass  # skip malformed rows (e.g. partial write at crash time)
    except Exception:
        pass
    return done


def process_sentences(sentences, mode):
    """Batch processing for TXT output mode. Returns (valid_sentences, invalid_sentences)."""
    valid_sentences = []
    invalid_sentences = []

    evaluate = partial(evaluate_sentence, mode=mode)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(evaluate, sentences), total=len(sentences), desc="Processing sentences"))

    for sentence, result in zip(sentences, results):
        if "Invalid" in result:
            invalid_sentences.append(sentence)
        elif "Valid" in result:
            valid_sentences.append(sentence)

    return valid_sentences, invalid_sentences


def process_sentences_csv(indexed_sentences, output_path, mode, workers):
    """CSV output mode with checkpoint/resume and incremental disk writes.

    indexed_sentences: list of (line_num, sentence) for all sentences to process.
    Already-completed line numbers are loaded from output_path and skipped.
    Results are written to output_path immediately as each thread completes,
    so a crash only loses the sentences that were mid-flight at that moment.
    """
    done_indices = load_checkpoint(output_path)
    pending = [(idx, s) for idx, s in indexed_sentences if idx not in done_indices]

    if done_indices:
        print(f"Resuming: {len(done_indices)} already done, {len(pending)} remaining.")

    if not pending:
        print("All sentences already processed.")
        return

    append_mode = len(done_indices) > 0
    evaluate = partial(evaluate_sentence, mode=mode)

    with IncrementalCsvWriter(output_path, append=append_mode) as writer:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Each sentence is submitted exactly once; the executor's internal
            # queue ensures each future is claimed by exactly one worker thread.
            for idx, sentence in pending:
                future = executor.submit(evaluate, sentence)
                futures[future] = (idx, sentence)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sentences"):
                idx, sentence = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error on line {idx} ('{sentence}'): {e}")
                    continue  # skip; will be retried on next resume

                label = classify_result(result)
                if label is not None:
                    writer.write_row(idx, sentence, label)
                else:
                    print(f"Unclassified response for line {idx}: {result!r}")


def run_detector(mode, input_path, output_path, limit=None, workers=5, use_csv=False):
    print("-" * 50)
    print("Weirdness Detector Started")
    print(f"Mode: {mode} | Model: {MODEL_NAME}")
    print("-" * 50)

    with open(input_path, "r") as f:
        all_sentences = f.read().splitlines()

    if limit is not None:
        all_sentences = all_sentences[:limit]

    if use_csv:
        indexed = list(enumerate(all_sentences))
        process_sentences_csv(indexed, output_path, mode, workers)
        done = load_checkpoint(output_path)
        print(f"Sentences written to CSV: {len(done)}")
    else:
        valid_sentences, invalid_sentences = process_sentences(all_sentences, mode)

        with open(output_path, "w") as f:
            f.write(f"Total sentences processed successfully: {len(valid_sentences) + len(invalid_sentences)}\n")
            f.write(f"Valid sentences: {len(valid_sentences)} \n")
            f.write(f"Invalid sentences: {len(invalid_sentences)} \n\n")

            f.write("---- VALID SENTENCES ----\n")
            f.writelines([f"{sentence}\n" for sentence in valid_sentences])
            f.write("\n---- INVALID SENTENCES ----\n")
            f.writelines([f"{sentence}\n" for sentence in invalid_sentences])

        print(f"Valid sentences: {len(valid_sentences)}")
        print(f"Invalid sentences: {len(invalid_sentences)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weirdness detector: classify sentences via Ollama as Valid/Invalid."
    )
    parser.add_argument(
        "input",
        help="Path to input text file (one sentence per line)."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama model name (default: {MODEL_NAME})."
    )
    parser.add_argument(
        "--mode",
        choices=["STRICT", "RELAXED"],
        default="STRICT",
        help="Prompt mode: STRICT or RELAXED (default: STRICT)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N sentences (default: all)."
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write output as CSV with checkpoint/resume support (line_num, sentence, label columns)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel worker threads (default: 5)."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the output file path. Default: {input_dir}/{stem}_checked.txt (or .csv with --csv)."
    )
    return parser.parse_args()


def main():
    global MODEL_NAME
    args = parse_args()

    if args.model:
        MODEL_NAME = args.model

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    stem = input_path.stem
    directory = input_path.parent

    if args.output:
        output_path = Path(args.output)
    elif args.csv:
        output_path = directory / f"{stem}_checked.csv"
    else:
        output_path = directory / f"{stem}_checked.txt"

    start_time = time.time()
    run_detector(
        mode=args.mode,
        input_path=input_path,
        output_path=output_path,
        limit=args.limit,
        workers=args.workers,
        use_csv=args.csv,
    )
    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
