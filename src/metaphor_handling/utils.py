
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
import torch
import numpy as np
import subprocess
import time
import re
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


# -------------------------------------
# CONSTANTS
# -------------------------------------
TRANS_PROMPT_INITIAL = (
    'You are a language model that helps rephrase metaphorical sentences into '
    'literal, non-metaphorical language.\n\n'
    'Given a metaphorical sentence, return a JSON object with the following fields:\n'
    '- "original": the original metaphorical sentence\n'
    '- "reworded": a literal, rephrased version of the sentence that keeps the original meaning as closely as possible\n\n'
    'Here is the sentence:\n'
    '"{sentence}"'
)

TRANS_PROMPT_FOLLOW_ON = (
    'The reworded sentence you provided was too dissimilar from the original metaphorical sentence in terms of meaning.\n\n'
    'Please try again. Make sure the new sentence closely preserves the original meaning, but removes the metaphor.\n\n'
    'Return a JSON object with the following fields:\n'
    '- "original": the original metaphorical sentence\n'
    '- "reworded": a literal, rephrased version of the sentence\n\n'
    'Here is the original metaphorical sentence again:\n'
    '"{sentence}"'
)


WSD_PROMPT_OLD = (
    "Sentence: \"{sentence}\"\n"
    "Word: \"{word}\"\n"
    "Possible senses (sense_key - definition):\n"
    "{senses}"
    "Question: Based on the context of the sentence, identify the sense in which the word is being used. "
    "If none of the provided senses match well, leave the 'sense_key' and 'definition' fields blank. "
    "Return your answer as a JSON object with exactly the following keys:\n\n"
    "  'word': the target word,\n"
    "  'sense_key': the WordNet sense key that best matches the usage (or an empty string if no good sense is found),\n"
    "  'definition': the definition for that WordNet sense (or an empty string if no good sense is found),\n"
    "  'reasoning': your explanation of why this sense was selected (or why no good sense was found).\n\n"
    "Please output only the JSON object and nothing else."
)

WSD_PROMPT = (
    "Sentence: \"{sentence}\"\n"
    "Target word: \"{word}\"\n\n"
    "Possible senses (each has an offset and definition):\n"
    "{senses}\n"
    "Instructions:\n"
    # "- Choose the correct sense **by selecting one of the exact synset offsets listed above** (e.g., '00017237').\n"   # THIS IS CAUSING TROUBLE, IT PICKS THE EXAMPLE!!!
    "- Choose the correct sense **by selecting one of the exact synset offsets listed above**.\n"
    "- **Do not modify, abbreviate, or otherwise change the provided offsets.**\n"
    "- If none of the provided senses match well, leave 'offset' and 'definition' blank (empty string).\n\n"
    "Respond in the following JSON format (and output **only** the JSON, without any extra text):\n\n"
    "{{\n"
    "  \"word\": \"{word}\",\n"
    "  \"offset\": \"<exact synset offset from above or empty string>\",\n"
    "  \"definition\": \"<definition of that sense or empty string>\",\n"
    "  \"reasoning\": \"<brief explanation for your choice>\"\n"
    "}}"
)






# -------------------------
# ROUGE-L Similarity Function
# -------------------------
def compute_rouge(reference, candidate, **kwargs):
    """
    Compute a simple ROUGE-L F-measure between the reference and candidate sentences.
    This function uses the length of the longest common subsequence (LCS) to compute scores.
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    m, n = len(ref_tokens), len(cand_tokens)
    
    # Create a DP table to compute the LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == cand_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    
    # Calculate precision and recall based on LCS
    precision = lcs / n if n > 0 else 0
    recall = lcs / m if m > 0 else 0
    
    if precision + recall == 0:
        f_measure = 0.0
    else:
        f_measure = (2 * precision * recall) / (precision + recall)
    
    return f_measure

# -------------------------
# BLEU Similarity Function
# -------------------------
def compute_bleu(reference, candidate, **kwargs):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    # weights=(0,1) means “0% unigram, 100% bigram”
    weights = (0, 1)
    # smoothing to avoid zero on short sentences
    smoother = SmoothingFunction().method4
    return sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=weights,
        smoothing_function=smoother
    )

# -------------------------
# BERT Cosine Similarity Function
# -------------------------
class BertEmbedder:
    def __init__(self, model_name="distilbert-base-uncased", device=None, use_mean_pooling=True):
        """
        If 'sentence-transformers' in model_name, will use SentenceTransformer.
        Otherwise, uses HuggingFace Transformers model with optional mean pooling.
        """
        self.use_sentence_transformers = model_name.startswith("sentence-transformers") or "MiniLM" in model_name

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mean_pooling = use_mean_pooling
        self._cache = {}

        if self.use_sentence_transformers:
            self.model = SentenceTransformer(model_name, device=str(self.device))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)

    def encode(self, text):
        if text in self._cache:
            return self._cache[text]

        if self.use_sentence_transformers:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                outputs = self.model(**inputs)

                if self.use_mean_pooling:
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                    sum_mask = input_mask_expanded.sum(dim=1)
                    embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()
                else:
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        self._cache[text] = embedding
        return embedding

    def encode_batch(self, texts):
        """
        Encode a batch of sentences and return their embeddings.
        Uses the cache: if a sentence has already been encoded, its cached value is used.
        """
        # List to store texts that need encoding.
        to_encode = []
        indices = []  # Store indices of texts that are not cached.
        embeddings = [None] * len(texts)
        
        # Look for cached embeddings.
        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings[i] = self._cache[text]
            else:
                indices.append(i)
                to_encode.append(text)
        
        if to_encode:
            with torch.no_grad():
                inputs = self.tokenizer(to_encode, return_tensors="pt", truncation=True, padding=True)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            # Save newly computed embeddings and place them in the result list.
            for count, idx in enumerate(indices):
                embeddings[idx] = batch_embeddings[count]
                self._cache[texts[idx]] = batch_embeddings[count]
                
        return np.array(embeddings)

# Initialize a global BERT embedder instance
#bert_embedder = BertEmbedder()

def compute_bert_cosine(sentence1, sentence2, bert_embedder):
    """
    Compute the cosine similarity between the BERT embeddings of two sentences.
    """
    emb1 = bert_embedder.encode(sentence1)
    emb2 = bert_embedder.encode(sentence2)
    # Compute cosine similarity using numpy
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return cosine_sim

# -------------------------
# Master Similarity Function
# -------------------------
def compute_similarity(metric, sentence1, sentence2, **kwargs):
    """
    Compute similarity between two sentences based on the specified metric.
    
    Parameters:
        metric (str): One of "rouge", "bleu", or "bert_cosine".
        sentence1 (str): The first sentence (often the reference).
        sentence2 (str): The second sentence (often the candidate).
        **kwargs: Additional arguments to be passed to the similarity function.
    
    Returns:
        float: The computed similarity score.
        
    Raises:
        ValueError: If an unknown metric is provided.
    """
    similarity_functions = {
        "rouge": compute_rouge,
        "bleu": compute_bleu,
        "bert_cosine": compute_bert_cosine
    }
    
    if metric not in similarity_functions:
        raise ValueError(f"Unknown similarity metric: {metric}. Available metrics: {list(similarity_functions.keys())}")
    
    return similarity_functions[metric](sentence1, sentence2, **kwargs)

# Uncomment the next line if you haven't already downloaded the WordNet corpus.

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

def composite_score(
    bert_cosine_score: float,
    bleu_score: float,
    rouge_score: float,
    corrected_proportion: float,
    reduction_ratio: float,
    weights: tuple[float, float, float, float, float] = (0.15, 0.15, 0.10, 0.50, 0.10),
) -> float:
    """
    Compute a composite score from five metrics, allowing the reduction_ratio to
    go as low as -1 (i.e. more flagged words in translation than original).

    Args:
      bert_cosine_score: ∈ [0,1]
      bleu_score:          ∈ [0,1]
      rouge_score:         ∈ [0,1]
      corrected_proportion:∈ [0,1]
      reduction_ratio:     ∈ (−∞,1], will be clamped to [−1,1]
      weights: (w_bert, w_bleu, w_rouge, w_corr, w_red) summing to 1

    Returns:
      float: weighted sum, with negative reduction_ratios penalized down to −1.
    """
    w_bert, w_bleu, w_rouge, w_corr, w_red = weights
    assert abs(sum(weights) - 1) < 1e-6, "Weights must sum to 1"

    # If no metaphors were flagged originally, use corrected_proportion as reduction
    if reduction_ratio is None:
        print('PERFECT CORRECTIONS, ADDING MORE WEIGHT TO METAPHOR CORRECTION')
        norm_reduction = corrected_proportion
    else:
        # clamp into [−1, 1]
        norm_reduction = max(-1.0, min(reduction_ratio, 1.0))

    return (
        w_bert  * bert_cosine_score
      + w_bleu  * bleu_score
      + w_rouge * rouge_score
      + w_corr  * corrected_proportion
      + w_red   * norm_reduction
    )



def get_wordnet_senses(word):
    """
    Given a word, this function retrieves all its WordNet senses (synsets)
    and returns a set of synset names.
    
    Parameters:
        word (str): The input word.
    
    Returns:
        set: A set containing the names of each synset associated with the word.
    """
    # Retrieve all synsets for the word and extract their names.
    senses = {synset.name(): synset.definition() for synset in wn.synsets(word)}

    return senses

def get_wn_pos_tags(sentence):
    """
    Retrieves the part-of-speech tags for all occurrences of a given word in a sentence.

    Parameters:
        word (str): The target word to look for.
        sentence (str): The sentence in which to find the word.
    
    Returns:
        list: A list of POS tags corresponding to each occurrence of the word in the sentence.
    """
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)

    wn_tagged_tokens = []
    for word, pos in tagged_tokens:
        wn_tagged_tokens.append((word, penn_to_wordnet(pos)))

    return wn_tagged_tokens

def get_filtered_wordnet_senses(word, pos):
    """
    Retrieves WordNet senses for a given word that match the specified part-of-speech (POS) tag.

    This function iterates over all synsets (senses) of the given word in WordNet and filters them
    based on the provided POS tag. It returns a dictionary where each key is the synset name and each
    value is the corresponding definition.

    Parameters:
        word (str): The target word for which to retrieve senses.
        pos (str): The WordNet part-of-speech tag to filter by (e.g., 'n' for nouns, 'v' for verbs,
                   'a' or 's' for adjectives, 'r' for adverbs).

    Returns:
        dict: A dictionary mapping synset names (str) to their definitions (str) for senses that match
              the specified POS.
    """
    synset_filtered = {}
    for syn in wn.synsets(word):
        if syn.pos() == pos:
            #synset_filtered[syn.name()] = syn.definition()
            synset_filtered[syn.offset()] = syn.definition()
    return synset_filtered

def get_wsd_prompt(sentence, word, senses_dict):
    """
    Constructs the prompt for word sense disambiguation.

    Parameters:
        sentence (str): The context sentence.
        word (str): The target word.
        senses_dict (dict): A dictionary mapping sense keys to definitions.

    Returns:
        str: The formatted prompt with all variables inserted.
    """
    senses_text = ""
    for sense_key, definition in senses_dict.items():
        senses_text += f"- {sense_key}: {definition}\n"
    
    # Use the .format() method to fill in the placeholders.
    return WSD_PROMPT.format(sentence=sentence, word=word, senses=senses_text)


def gen_translation_prompt(
    sentence: str,
    flagged_words: list[str] | None = None,
    follow_on: bool = False,
    sim_failed: bool = False,
    metaphor_failed: bool = False
) -> str:
    """
    Constructs a prompt for zero-shot rephrasing of a metaphorical sentence
    into basic, literal English, or a context-aware follow-on prompt if the first try failed
    due to similarity, metaphor retention, or both.
    """

    # Build an optional note for flagged words
    flagged_section = ""
    if flagged_words:
        flagged_list = ", ".join(f'"{w}"' for w in flagged_words)
        flagged_section = (
            f"\n\nNote: The following word(s) were flagged as possibly metaphorical: {flagged_list}.\n"
            "Only change these words if you agree they are metaphorical.\n"
        )

    if follow_on:
        # Context-aware follow-on instruction based on failure reason
        reasons = []
        if sim_failed:
            reasons.append("was too dissimilar from the original in meaning")
        if metaphor_failed:
            reasons.append("still included metaphorical language")
        reason_text = " and ".join(reasons)

        follow_on_text = (
            f"The previous reworded sentence {reason_text}.\n\n"
            "Please try again. Make sure the new sentence closely preserves the original meaning, "
            "removes any metaphors, and uses very plain, simple language."
        )

        return (
            f"{follow_on_text}\n\n"
            f"{flagged_section}\n"
            "Return a JSON object with the following fields:\n"
            "- \"original\": the original metaphorical sentence\n"
            "- \"reworded\": a literal, rephrased version of the sentence\n\n"
            f"Here is the original metaphorical sentence again:\n\"{sentence}\""
        )

    # Zero-shot prompt
    return (
        "You are a model that rewrites metaphorical sentences into basic, literal English.\n"
        "Use very plain, simple language — like something a 3rd grader could read.\n"
        "Keep the sentence structure almost the same if possible.\n"
        "Only change words that are metaphorical."
        f"{flagged_section}\n"
        f"Sentence:\n\"{sentence}\"\n\n"
        "Return a JSON object with the following fields:\n"
        "- \"original\": the original sentence\n"
        "- \"reworded\": the plain version of the sentence\n"
        "Make sure the JSON is properly formatted."
    )


def gen_translation_prompt2(
    sentence: str,
    flagged_words: list[str] | None = None,
    follow_on: bool = False
) -> str:
    """
    Constructs a prompt for zero-shot rephrasing of a metaphorical sentence
    into basic, literal English, or a follow-on prompt if the first try was too dissimilar.
    """

    # Build an optional note for flagged words
    flagged_section = ""
    if flagged_words:
        flagged_list = ", ".join(f'"{w}"' for w in flagged_words)
        flagged_section = (
            f"\n\nNote: The following word(s) were flagged as possibly metaphorical: {flagged_list}.\n"
            "Only change these words if you agree they are metaphorical.\n"
        )

    if follow_on:
        # Inline follow-on instruction
        follow_on_text = (
            "The reworded sentence you provided was too dissimilar from the original metaphorical sentence in terms of meaning.\n\n"
            "Please try again. Make sure the new sentence closely preserves the original meaning, but removes the metaphor."
        )
        return (
            f"{follow_on_text}\n\n"
            f"{flagged_section}\n"
            "Return a JSON object with the following fields:\n"
            "- \"original\": the original metaphorical sentence\n"
            "- \"reworded\": a literal, rephrased version of the sentence\n\n"
            f"Here is the original metaphorical sentence again:\n\"{sentence}\""
        )

    # Zero-shot prompt
    return (
        "You are a model that rewrites metaphorical sentences into basic, literal English.\n"
        "Use very plain, simple language — like something a 3rd grader could read.\n"
        "Keep the sentence structure almost the same if possible.\n"
        "Only change words that are metaphorical."
        f"{flagged_section}\n"
        f"Sentence:\n\"{sentence}\"\n\n"
        "Return a JSON object with the following fields:\n"
        "- \"original\": the original sentence\n"
        "- \"reworded\": the plain version of the sentence\n"
        "Make sure the JSON is properly formatted."
    )


def gen_translation_prompt1(
    sentence: str,
    flagged_words: list[str] | None = None
) -> str:
    """
    Constructs a prompt for zero-shot rephrasing of a metaphorical sentence
    into basic, literal English.
    """

    # Build an optional note for flagged words
    flagged_section = ""
    if flagged_words:
        flagged_list = ", ".join(f'"{w}"' for w in flagged_words)
        flagged_section = (
            f"\n\nNote: The following word(s) were flagged as possibly metaphorical: {flagged_list}.\n"
            "Only change these words if you agree they are metaphorical.\n"
        )

    # Core instructions (third-grader level, maintain structure, only change metaphors)
    prompt = (
        "You are a model that rewrites metaphorical sentences into basic, literal English.\n"
        "Use very plain, simple language — like something a 3rd grader could read.\n"
        "Keep the sentence structure almost the same if possible.\n"
        "Only change words that are metaphorical."
        f"{flagged_section}\n"
        f"Sentence:\n\"{sentence}\"\n\n"
        "Return a JSON object with the following fields:\n"
        '- "original": the original sentence\n'
        '- "reworded": the plain version of the sentence\n'
        "Make sure the JSON is properly formatted."
    )

    return prompt


def gen_batch_translation_prompt(
    sentence: str,
    batch_size: int,
    flagged_words: list[str] | None = None,
    follow_on: bool = False
) -> str:
    """
    Constructs a prompt for rephrasing a metaphorical sentence into literal, basic language.
    """

    flagged_section = ""
    if flagged_words:
        flagged_list = ", ".join(f'"{w}"' for w in flagged_words)
        flagged_section = (
            f"\n\nNote: The following word(s) were flagged as possibly metaphorical: {flagged_list}.\n"
            "Only change these words if you agree they are metaphorical.\n"
        )

    translations = "\n".join(
        f'- "{i}": a plain version of the sentence with simple words and no metaphors (variation {i}).'
        for i in range(1, batch_size + 1)
    )

    core_instruction = (
        "You are a model that rewrites metaphorical sentences into basic, literal English.\n"
        "Use very plain, simple language — like something a 3rd grader could read.\n"
        "Keep the sentence structure almost the same if possible.\n"
        "Only change words that are metaphorical.\n"
        f"{flagged_section}\n"
        f"Sentence:\n\"{sentence}\"\n\n"
        "Return a JSON object with the following fields:\n"
        f'- "original": the original sentence\n{translations}\n'
        "- Important: Each version should be different, but all must stay simple and literal."
    )

    if follow_on:
        return (
            "The previous responses were too complex or still metaphorical.\n"
            "Please try again and follow the instructions more closely.\n\n"
            + core_instruction
        )
    return core_instruction


def gen_batch_translation_prompt1(
    sentence: str,
    batch_size: int,
    flagged_words: list[str] | None = None,
    follow_on: bool = False
) -> str:
    """
    Constructs a prompt for batch literal rephrasings, optionally including
    detector-flagged words that may still be metaphorical.

    Parameters:
        sentence (str): The original metaphorical sentence.
        batch_size (int): How many reworded variations to generate.
        flagged_words (list[str] | None): Words flagged by the detector.
        follow_on (bool): If True, use a follow-on prompt for revising unsatisfactory outputs.

    Returns:
        str: A formatted instruction prompt.
    """
    # Build the list of translation items
    translations_instructions = ""
    for i in range(1, batch_size + 1):
        translations_instructions += (
            f'- "{i}": a literal, rephrased version of the sentence that closely preserves '
            f'the original meaning (variation {i}).\n'
        )
    extra_instruction = (
        "\n- **Important:** Each reworded sentence must be unique from the others, "
        "with distinct phrasing."
    )

    # Build the flagged-words section
    if flagged_words:
        flagged_list = ", ".join(f'"{w}"' for w in flagged_words)
        flagged_section = (
            "Note: our metaphor detector flagged these words as possibly metaphorical: "
            f"{flagged_list}. "
            "The detector has only ~77% word-level F1 and sometimes misflags common words like 'to' or 'and'. "
            "Only revise your rewordings if you agree a flagged word is truly metaphorical.\n\n"
        )
    else:
        flagged_section = ""

    # Construct initial vs. follow-on prompts
    if not follow_on:
        prompt = (
            "You are a language model that helps rephrase metaphorical sentences into "
            "literal, non-metaphorical language.\n\n"
            + "The sentence is:\n"
            f'"{sentence}"\n'
            f"{flagged_section}"
            "Given a metaphorical sentence, return a JSON object with the following fields:\n"
            '- "original": the original metaphorical sentence\n'
            + translations_instructions
            + extra_instruction
            + "\n\nHere is the sentence again:\n"
            f'"{sentence}"'
        )
    else:
        prompt = (
            "The translations you provided were not literal enough or missed some metaphors.\n\n"
            + "Here is the original sentence again:\n"
            f'"{sentence}"\n'
            f"{flagged_section}"
            "Please try again. Make sure each reworded sentence closely preserves the original meaning, "
            "but removes any metaphorical phrasing.\n\n"
            "Return a JSON object with the following fields:\n"
            '- "original": the original metaphorical sentence\n'
            + translations_instructions
            + extra_instruction
            # + "\n\nHere is the original sentence again:\n"
            # f'"{sentence}"'
        )

    return prompt




def gen_retry_prompt(sentence, failed_translations, batch_size, follow_on=False):
    """
    Constructs a prompt for the retry translation task that asks for multiple literal rephrasings,
    including the failed translations (which were still metaphorical) as examples.

    Parameters:
        sentence (str): The original metaphorical sentence.
        failed_translations (list of str): Translations previously generated that were still metaphorical.
        batch_size (int): The number of new reworded variations to request.
        follow_on (bool): If True, provide stronger instructions for preserving meaning.

    Returns:
        str: A formatted prompt string.
    """
    # Format the failed examples


    # build each section
    if failed_translations['sim']:
        failed_sim = (
            "The following translations were generated previously and got rid of the metaphor, "
            "but are not similar enough to the original sentence:\n"
            + "\n".join(
                f"  {i}. \"{f}\""
                for i, f in enumerate(failed_translations['sim'], start=1)
            )
        )
    else:
        failed_sim = ""

    if failed_translations['met']:
        failed_met = (
            "The following translations were generated previously are similar enough to the original sentence, "
            "but still contain metaphorical language:\n"
            + "\n".join(
                f"  {i}. \"{f}\""
                for i, f in enumerate(failed_translations['met'], start=1)
            )
        )
    else:
        failed_met = ""

    if failed_translations['both']:
        failed_both = (
            "The following translations were generated previously and are not similar enough to the original sentence "
            "and also still contain metaphorical language:\n"
            + "\n".join(
                f"  {i}. \"{f}\""
                for i, f in enumerate(failed_translations['both'], start=1)
            )
        )
    else:
        failed_both = ""

    # combine only the non-empty sections
    sections = [sec for sec in (failed_sim, failed_met, failed_both) if sec]
    failed_examples = "\n\n".join(sections)

    

    # Create the rewording instructions
    translations_instructions = ""
    for i in range(1, batch_size + 1):
        translations_instructions += (
            f'- "{i}": a literal, rephrased version of the sentence that closely preserves '
            f'the original meaning (variation {i}).\n'
        )

    extra_instruction = (
        "\n- **Important:** Each reworded sentence must be unique from the others and the previously generated rewordings "
        "with distinct phrasing."
    )

    prompt = (
        "You are a language model that helps rephrase metaphorical sentences into literal, non-metaphorical language.\n\n"
        f"{failed_examples}\n"
        "Please rephrase the original sentence to eliminate all metaphors. Return a JSON object with the following fields:\n"
        '- "original": the original metaphorical sentence\n'
        + translations_instructions + extra_instruction + "\n\n"
        "Here is the original metaphorical sentence:\n"
        f"\"{sentence}\""
    )

    return prompt

def penn_to_wordnet(tag):
    """
    Maps Penn Treebank POS tags to WordNet POS tags.
    
    Parameters:
        tag (str): The Penn Treebank POS tag.
        
    Returns:
        str or None: The corresponding WordNet POS tag (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV),
                     or None if no appropriate mapping exists.
    """
    if tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('J'):
        # Penn Treebank uses 'JJ', 'JJR', 'JJS' for adjectives.
        return wn.ADJ
    elif tag.startswith('R'):
        # Penn Treebank uses 'RB', 'RBR', 'RBS' for adverbs.
        return wn.ADV
    else:
        return None

def start_ollama_server():
    try:
        # Execute the bash script in the background.
        subprocess.Popen(["bash", "/home/jarrad.singley/data/workspace/sumonlp/src/metaphor_handling/ollama.sh"])
        print("Bash script executed. Waiting for Ollama server to start...")
        # Wait for the server to start (adjust the sleep time if needed).
        time.sleep(6)
    except Exception as e:
        print(f"Error executing bash script: {e}")
        return


def build_word_index_dict(sentence):
    """
    Splits sentence on whitespace and returns a dict mapping
    the character index of each word's first letter to the word itself.
    """
    word_index = {}
    # \S+ matches runs of non‑whitespace chars
    for m in re.finditer(r'\S+', sentence):
        word_index[m.start()] = m.group()
    return word_index


def fix_metaphor_words_dict(sentence_dict):
    sen = sentence_dict.get('sentence', '')
    sentence_metaphors = sentence_dict.get('sentence_metaphors', {})

    # Build index→word map (splitting on whitespace)
    all_words = build_word_index_dict(sen)
    fixed    = {}
    consumed = set()

    for sm in sorted(sentence_metaphors):
        if sm in consumed:
            continue

        # 1) Exact‐start match
        if sm in all_words:
            w_start, word = sm, all_words[sm]
        else:
            # 2) Fallback: find the word that encloses this fragment
            for w_start, word in all_words.items():
                if w_start < sm < w_start + len(word):
                    break
            else:
                # no enclosing word found (should be rare)
                continue

        # Clean punctuation
        clean = word.strip(string.punctuation)
        fixed[w_start] = clean

        # Mark all sub‐fragments inside as consumed
        w_end = w_start + len(word)
        for other in sentence_metaphors:
            if w_start < other < w_end:
                consumed.add(other)

    # DEBUG if something still weird:
    if sentence_metaphors and not fixed:
        print("SOMETHING WEIRD IS HAPPENING!!!")
        print("Sentence:", sen)
        print("Raw hits:", sentence_metaphors)
        print("Fixed:", fixed)

    return fixed

def count_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

if __name__ == "__main__":
    # Example usage:
    # sentence = "I need to go to the bank to deposit money, and the bank was crowded."
    # word = "bank"
    # pos_tags = get_wn_pos_tags(sentence)
    # print(f"WordNet POS tags for '{sentence}' :")
    # print(pos_tags)


    # senses = get_wordnet_senses(word)
    # print(f"Senses for '{word}':")
    # for sense, definition in senses.items():
    #     print(sense, definition)

    # print(get_filtered_wordnet_senses(word))

    s1 = "You must adhere to the rules"
    c1 = "You must follow the rules closely"
    c2 = "You must stick to the rules"

    print(compute_bleu(s1, c1))
    print(compute_bleu(s1, c2))