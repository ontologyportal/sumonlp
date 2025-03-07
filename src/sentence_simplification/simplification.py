import os
import glob
import datetime
import ollama
import time
import random
from complexity import *
import stanza
import tqdm
import re
from util import *
from sentence_transformers import SentenceTransformer, util
import spacy 
nlp = spacy.load("en_core_web_sm")
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sentence_simplification')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sentence_simplification/preprocessing')))

from asset_embeddings import get_sentence_pairs, get_custom_sentence_pairs
from complexity import determine_complexity

EXAMPLE_SENTENCES_TYPES = ["dynamic_similarity", "dynamic_tree", "static", "random", "custom"]
DEFAULT_MODEL = 'llama3.1:8b-instruct-fp16'


def call_ollama(prompt, model_type):
    """Call the Ollama model with the given prompt and model type."""
    try:
        # Use the ollama library to send the prompt to the model
        # temperature to 0 means there is no creativity, and responses are deterministic.
        response = ollama.generate(model=model_type, prompt=prompt, options={"temperature": 0})
        message = response["response"]
        with open("ollama_log.txt", "a") as log:
            log.write(f'Prompt: {prompt}\n')
            log.write(f'Output: {message}\n\n')
        return(message)
    
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def simplify_sentence(sentence, model=DEFAULT_MODEL, context_size=3, context_type='custom', complexity_filter=False):
    '''Simplifies a sentence using the given model and context settings'''

    # Check if the sentence is a conjecture
    if '?' in sentence:
        query = f"Is the following sentence a question? '{sentence}' Answer 'Yes' or 'No'."
        response = call_ollama(query, model)
        if 'yes' in response.lower():
            return sentence, False
    
    if complexity_filter:
        complex, complexity_dict = determine_complexity(sentence)
        if not complex:
            return sentence, False
    if context_size > 0 and context_type in [t for t in EXAMPLE_SENTENCES_TYPES if t != 'custom']:
        sentence_pairs = get_sentence_pairs(sentence, context_size, context_type)
    elif context_size > 0 and context_type == 'custom':
        sentence_pairs = get_custom_sentence_pairs(context_size)
    else:
        sentence_pairs = None

    context_examples = '\n'.join(
        f'Original: {orig}\nResponse: {simp}'
        for orig, simp in sentence_pairs
    ) if sentence_pairs else ''

    context = (
        'Your task is to simplify a sentence. Do not add redundant sentences. '
        'If the sentence is already simple, LEAVE IT UNCHANGED. '
        'Do not simplify verbs or nouns, LEAVE VERBS AND NOUNS ALONE. '
        'Here are some examples:\n'
        f'{context_examples}\n' if context_examples else ''
    )

    query = (
        f'{context}Convert the following sentence. Respond in the format '
        f"'Since <reason>, I will <action>. The converted version is: <converted version>'\n"
        f'Original: {sentence.strip()}\nResponse:'
    )

    print(f"Prompt: {query}")
    response = ollama.generate(model, prompt=query, options={"temperature": 0})

    message = response['response'].replace('\n', ' ')
    
    with open("ollama_log.txt", "a") as log:
        log.write(f'Simplify prompt: {query}\n')
        log.write(f'Simplify output: {message}\n\n')

    message = message.split("The converted version is: ")[-1].strip()



    print(f'original sentence: {sentence}')
    print(f'simplified sentences: {message}')


    if message == sentence:
        return message, False
    return message, True


def ollama_hallucination_check(original: str, simplified: str, model: str = DEFAULT_MODEL):
    """
    Uses Ollama to check if the simplified sentence hallucinates extra details or removes key information.

    Args:
        original (str): The original sentence.
        simplified (str): The simplified version of the sentence.
        model (str): The Ollama model to use (default: "llama3").

    Returns:
        dict: A dictionary with detected hallucination issues.
    """
    
    prompt = f"""
    You are evaluating sentence simplifications for language-to-logic translation.
    
    Identify any hallucinations (extra details not found in the original), omissions (important missing information), or meaning changes.
    Small changes between verbs that still mean the same thing are okay, as long as they are logically equivalent. Do not penalize split sentences, as long as the meaning is preserved. Splitting is one of the main purposes of simplification. Pronouns will be resolved in a later step, so do not worry about them here.

    Respond in the format:
    'Since <reasoning>, I have determined:
        - Extra Information: <describe> or None
        - Missing Information: <describe> or None
        - Meaning Change: <describe> or None
        - Gained Complexity: <Yes/No> (implying the simplified version is more complex than original)
        - Recommendation: <recommendation> or 'Enthusiastically accept.' if no issues are found.
    
    Compare the following two sentences:
    
    Original: "{original}"
    Simplified: "{simplified}"

    """

    response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0})
    analysis = response["response"]

    with open("ollama_log.txt", "a") as log:
        log.write(f"Hallucination prompt: {prompt}\n")
        log.write(f"Hallucination output: {analysis}\n\n")

    issues = {}
    for line in analysis.split("\n"):
        if line.startswith("- Extra Information:"):
            issues["extra_info"] = line.replace("- Extra Information:", "").strip() if "None" not in line else None
        elif line.startswith("- Missing Information:"):
            issues["missing_info"] = line.replace("- Missing Information:", "").strip() if "None" not in line else None
        elif line.startswith("- Meaning Change:"):
            issues["meaning_change"] = line.replace("- Meaning Change:", "").strip() if "None" not in line else None
        elif line.startswith("- Gained Complexity:"):
            issues["gained_complexity"] = line.replace("- Gained Complexity:", "").strip() if "No" not in line else None
        elif line.startswith("- Recommendation:"):
            issues["recommendation"] = line.replace("- Recommendation:", "").strip()

    if 'enthusiastically accept' in issues.get('recommendation', '').lower(): 
        return {}

    issues = {key: value for key, value in issues.items() if value is not None}


    return issues

def hallucination_check(original: str, simplified: str, threshold: float = 0.8) -> dict:
    """
    Check if the simplified sentence is hallucinated by comparing it to the original sentence.

    Args:
        original (str): The original sentence.
        simplified (str): The sentence generated by the model.
        threshold (float): The minimum similarity score to consider the sentence as valid.

    Returns:
        dict: A dictionary with detected hallucination issues.
    """
    issues = {}

    # Compute semantic similarity
    similarity_score = util.pytorch_cos_sim(
        similarity_model.encode(original, convert_to_tensor=True),
        similarity_model.encode(simplified, convert_to_tensor=True)
    ).item()

    if similarity_score < threshold:
        issues["semantic_deviation"] = f"Similarity too low ({similarity_score:.2f})"

    # Named Entity Consistency Check
    original_entities = {ent.text.lower() for ent in nlp(original).ents}
    simplified_entities = {ent.text.lower() for ent in nlp(simplified).ents}
    print(f'Original entities: {original_entities}')
    print(f'Simplified entities: {simplified_entities}')

    extra_entities = simplified_entities - original_entities
    missing_entities = original_entities - simplified_entities

    if extra_entities:
        issues["extra_entities"] = f"New entities added: {extra_entities}"
    if missing_entities:
        issues["missing_entities"] = f"Entities missing: {missing_entities}"

    # Verb & Action Matching
    original_verbs = {token.lemma_ for token in nlp(original) if token.pos_ == "VERB"}
    simplified_verbs = {token.lemma_ for token in nlp(simplified) if token.pos_ == "VERB"}
    print(f'Original verbs: {original_verbs}')
    print(f'Simplified verbs: {simplified_verbs}')

    extra_verbs = simplified_verbs - original_verbs
    missing_verbs = original_verbs - simplified_verbs

    if extra_verbs:
        issues["extra_actions"] = f"Extra actions introduced: {extra_verbs}"
    if missing_verbs:
        issues["missing_actions"] = f"Key actions removed: {missing_verbs}"

    return issues

def validate_response(original, simplified):
    '''Check if the simplified sentence is valid by comparing it to the original sentence.'''
    print(f"Original: {original}")
    print(f"Simplified: {simplified}")

    issues = {}

    # check if there is extra information added to the simplified version (hallucination)
    issues.update(ollama_hallucination_check(original, simplified))

    # check if there is a note added to the simplified version (colon, parentheses, 'note')
    original = original.lower()
    simplified = simplified.lower()
    markers = ['(', ':', 'note']

    # Check if any of these markers exist in the original and simplified versions
    original_flags = {marker: marker in original for marker in markers}
    simplified_flags = {marker: marker in simplified for marker in markers}

    # Check if a marker appears in the simplified version but not in the original
    extra_notes = [marker for marker in markers if not original_flags[marker] and simplified_flags[marker]]

    if extra_notes:
        issues['explanation found'] = extra_notes

    print(f"Issues: {issues}")
    
    return issues



def simplify_file(input_file, output_file, model_type):
    ''' Simplify a file using the provided tokenizer and model. Flows through the following steps:
        1. check if sentence is complex and simplifies it if it is.
        2. checks if sentence contains pronouns and replaces them with accurate nouns if possible.
        3. joins all sentences and uses stanza to separate sentences to write onto individual lines of output. '''

    simplified_sentences = []  

    with open("ollama_log.txt", "w") as log:   # clear log file
        log.write("")

    with open(input_file, 'r') as f:
        lines = [re.sub(r' \,', ',', re.sub(r' \.', '.', line)) for line in f]
    
    for sentence in tqdm.tqdm(lines):

        with open("ollama_log.txt", "a") as log:
            log.write(f"Simplifying: {sentence}\n")

        sentence = sentence.strip()

        simplified, simplified_flag = simplify_sentence(sentence, model=model_type)
        if not simplified_flag:
            issues = {}
        else:
            issues = validate_response(sentence, simplified)
        issues_text = "\n".join(f"- {key}: {value}" for key, value in issues.items())
        tries = 0
        while issues != {} and tries < 3 and simplified != sentence:  # try to simplify again if issues are found
            prompt = f'In simplifying the following sentence, issues were identified\nPlease correct the simplification. Respond with just the simplified version.\nOriginal sentence: {sentence}\nOriginal simplification: {simplified}\nIssues with original simplification: {issues_text}\nOriginal sentence: {sentence}\nFixed simplified:'
            simplified = call_ollama(prompt, DEFAULT_MODEL)
            issues = validate_response(sentence, simplified)
            issues_text = "\n".join(f"- {key}: {value}" for key, value in issues.items())
            tries += 1



        pronouns = False
        if check_pronouns_ollama(simplified, model_type):
            pronouns = True
        with open("ollama_log.txt", "a") as log:
            log.write(f"Sentence: {simplified}\nPronouns: {pronouns}\n\n")
        if pronouns:
            simplified = remove_pronouns(simplified, model_type)

        simplified_sentences.append(simplified)
    
    #join all sentences
    simplified_text = ' '.join(simplified_sentences)
    simplified_sentences = split_sentences(simplified_text)

    with open(output_file, 'w') as outfile:
        for sentence in simplified_sentences:
            outfile.write(sentence + '\n')
        
            

def remove_pronouns(sentence, model):
    ''' 
    Prompts the passed model to perform coreference resolution, requiring explicit reasoning before resolving pronouns.
    Ensures output follows the strict format: "Since <reasoning>, I will <action>. The resolved output is: <sentences>"
    '''

    examples = """
    Here are some examples to help you understand pronoun resolution. A pronoun is considered resolvable if it can be replaced with a specific noun or proper noun in the sentence(s).

    1. 'Sarah loves painting. She spends hours on it.' 
    Response: Since 'she' refers to Sarah and 'it' refers to painting, I will replace 'she' with 'Sarah' and 'it' with 'painting'. The resolved output is: Sarah loves painting. Sarah spends hours painting.

    2. 'The dog barked at the cat. The cat ran away.' 
    Response: Since there are no pronouns in the sentence, I will leave it unchanged. The resolved output is: The dog barked at the cat. The cat ran away.

    3. 'John called Mike, but he didn’t answer.' 
    Response: Since 'he' refers to Mike, I will replace 'he' with 'Mike'. The resolved output is: John called Mike, but Mike didn’t answer.

    4. 'It is raining outside.' 
    Response: Since 'it' does not refer to anything specific, I will leave it unchanged. The resolved output is: It is raining outside.

    5. 'Alice met Bob. Alice gave Bob a book.' 
    Response: Since 'Alice' and 'Bob' are proper nouns, I will leave them unchanged. The resolved output is: Alice met Bob. Alice gave Bob a book.

    6. 'Tom saw Jerry. He waved at him.' 
    Response: Since 'he' refers to Tom, and 'him' refers to Jerry, I will replace 'he' with 'Tom' and 'him' with 'Jerry'. The resolved output is: Tom saw Jerry. Tom waved at Jerry.

    7. 'Lisa found her keys on the table.' 
    Response: Since 'her' refers to Lisa, I will replace 'her' with 'Lisa'. The resolved output is: Lisa found Lisa’s keys on the table.

    8. 'They say exercise is important.' 
    Response: Since 'they' does not refer to anything specific, I will leave it unchanged. The resolved output is: They say exercise is important.
    """

    prompt = f"""
    {examples}

    Now, perform coreference resolution on the following sentence(s). Follow these rules EXACTLY:
    - Begin with "Since <reasoning>, I will <action>."
    - End with "The resolved output is: <sentences>."
    - Do NOT include any additional text, punctuation, or formatting.
    - Leave the sentence(s) as unchanged as possible if pronouns cannot be resolved.

    Sentence(s): '{sentence}'
    Response:
    """

    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})

        response_text = response["message"]["content"].strip()

        # Extract the resolved output from the model response
        resolved_output = response_text.split("The resolved output is:", 1)[-1].strip()

        with open("ollama_log.txt", "a") as log:
            log.write(f"Pronoun resolution:\nOriginal: {sentence}\nResolved: {resolved_output}\n\n")

        return resolved_output
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None



def split_sentences(text):
    ''' Split a text into sentences using stanza. '''
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    doc = nlp(text)
    return [sentence.text for sentence in doc.sentences]