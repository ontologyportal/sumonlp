import json
import re
import random
import argparse

IGNORED_KEYWORDS = {
    "now", "instance", "subclass", "exists", "forall", "relation",
    "subrelation", "attribute", "and", "or", "xor", "not", "=>"
}


def load_ontology_terms(path):
    terms = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"ERROR: Malformed JSON on line {lineno} of {path}: {e}")
                    continue
                sumo_type = obj.get("sumo_type")
                if not sumo_type:
                    print(f"ERROR: Missing 'sumo_type' field on line {lineno} of {path}")
                    continue
                terms.append(sumo_type)
    except FileNotFoundError:
        print(f"ERROR: Ontology file not found: {path}")
        print("It can be generated from: sigmanlp semanticRetrieval module, "
              "using the buildingEmbeddingIndex sub-module")
        raise SystemExit(1)
    if not terms:
        print(f"ERROR: No valid entries found in {path}")
        print("It can be generated from: sigmanlp semanticRetrieval module, "
              "using the buildingEmbeddingIndex sub-module")
        raise SystemExit(1)
    return terms


def extract_sumo_terms(output_str):
    # Remove quoted strings
    cleaned = re.sub(r'"[^"]*"', '', output_str)
    # Tokenize on whitespace and parentheses
    tokens = re.split(r'[\s()]+', cleaned)
    terms = []
    for token in tokens:
        if not token:
            continue
        if token.startswith('?'):
            continue
        if token.lower() in IGNORED_KEYWORDS:
            continue
        terms.append(token)
    return terms


def transform_dataset(input_path, output_path, ontology_terms):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    for entry in data:
        original_input = entry["input"].strip()
        output = entry["output"].strip()

        real_terms = extract_sumo_terms(output)
        n_real = len(real_terms)

        # Target total list size: 1.5x to 3.5x the real term count
        target_total = random.uniform(1.5, 3.5) * n_real
        n_distractors = max(0, round(target_total) - n_real)

        distractors = random.sample(ontology_terms, min(n_distractors, len(ontology_terms)))

        combined = real_terms + distractors
        random.shuffle(combined)

        terms_list = ", ".join(combined)
        new_input = (
            f'Convert to SUO-KIF. Sentence: "{original_input}" '
            f'Only use terms from this list: [{terms_list}].'
        )

        new_data.append({
            "input": new_input,
            "output": output
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    print(f"Wrote {len(new_data)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add candidate SUMO terms (with random distractors) to JSON training dataset."
    )
    parser.add_argument("input_file", help="Path to the input JSON dataset")
    parser.add_argument("output_file", help="Path for the new output JSON dataset")
    parser.add_argument("ontology_file", help="Path to the ontology .jsonl file (ontology-export.jsonl)")
    args = parser.parse_args()

    ontology_terms = load_ontology_terms(args.ontology_file)
    print(f"Loaded {len(ontology_terms)} ontology terms from {args.ontology_file}")
    transform_dataset(args.input_file, args.output_file, ontology_terms)
