# Sentence Simplification Module

This directory contains the **sentence simplification** code used to convert complex sentences into simpler, logically explicit statements that are friendlier for downstream language-to-logic translation.

---

## Goals

- Split complex/compound sentences into **atomic propositions**.
- **Preserve meaning** (no added or missing facts).
- Prefer **explicit agents** (resolve pronouns where possible).
- Strip discourse fluff that doesn’t change logical content.
- Emit a **line-per-sentence** simplified file.

---

## Directory Contents

```
.
├─ examples/
│  ├─ custom.orig              # Sample complex sentences (one per line)
│  ├─ custom.simp              # Paired handcrafted simplifications (JSON)
├─ entry_point.sh              # Bash wrapper (starts Ollama, runs main.py)
├─ main.py                     # CLI entry point for batch simplification
├─ simplification.py           # Core LLM prompts, retries, checks, splitting
├─ util.py                     # Utilities (logging, formatting, helpers)
├─ complexity.py               # Complexity heuristics (optional filter, not implemented now)
└─ requirements.txt            # Python dependencies

```

---

## Install & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download Stanza English pipeline
python - <<'PY'
import stanza; stanza.download('en')
PY

# Ensure Ollama is installed and running
# https://ollama.com/
ollama pull llama3.1:8b-instruct-q8_0
```

---

## Usage

### Run via Bash wrapper

```bash
./entry_point.sh
```

### Run manually

```bash
python3 main.py input_ss.txt output_ss.txt llama3.1:8b-instruct-q8_0
```

- `input_ss.txt` → file with one sentence per line  
- `output_ss.txt` → where simplified sentences will be written  
- `model_type` → Ollama model (default: `llama3.1:8b-instruct-q8_0`)  

---

## Customization

### Add/Change Corpus Examples
- Add to **`custom.orig`** and **`custom.simp`** for training/evaluation context.  
- These pairs are pulled in by `get_custom_sentence_pairs`.

### Change Model
- Update `DEFAULT_MODEL` in **`simplification.py`**, or pass a model name as arg:
  ```bash
  python3 main.py input_ss.txt output_ss.txt llama3.2:70b
  ```

### Modify Prompt Behavior
- Edit **prompt templates** in `simplification.py`:
  - `INITIAL_PROMPT_TEMPLATE_LOGIC` → main simplification  
  - `HALLUCINATION_CHECK_PROMPT_TEMPLATE_LOGIC` → quality check  
  - `FIX_HALLUCINATION_ERRORS_PROMPT_TEMPLATE_LOGIC` → repair step  

---

## Notes

- **Logs**: `ollama_log.txt` stores prompts, responses, retries.  
