import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sentence_simplification')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sentence_simplification/preprocessing')))

import ollama
import time
import stanza
import tqdm
import json
import re
from util import *
from complexity import determine_complexity, check_pronouns_ollama

DEFAULT_MODEL = 'llama3.1:8b-instruct-q8_0'
CUSTOM_DIR = './examples/custom_l2l'

# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------
 
# 1. Simplification Prompt
INITIAL_PROMPT_TEMPLATE_LOGIC = """Your task is to simplify a sentence for language to logic translation. 

Key Guidelines:
    - Do not change the meaning of the sentence.
    - Avoid adding any new information not present in the original sentence.
    - Maximize splitting the sentence into multiple smaller sentences.

{context}

Simplify the following sentence. Return a JSON object in the following format:

{{
    "Original": "<original sentence>",
    "Explanation": "<explanation of simplification>",
    "Simple": "<simplified version>"
}}

Do not nest JSON objects.
Do not include arrays.
Do not escape single quotes with a backslash.
Only use double quotes (") for keys and string values.
Do not include markdown formatting (no triple backticks).

Here is the sentence to simplify: {sentence}"""

# 2. Improper JSON Prompt
IMPROPER_JSON_PROMPT_TEMPLATE_LOGIC = """Your task is to simplify a sentence for language to logic translation. 

Key Guidelines:
    - Do not change the meaning of the sentence.
    - Avoid adding any new information not present in the original sentence.
    - Maximize splitting the sentence into multiple smaller sentences.

{context}

Your last response did not include the correct structure. Please ensure you return a valid JSON object with the keys "Original", "Explanation", and "Simple". 
Do not escape single quotes with a backslash.
Only use double quotes (") for keys and string values.
Do not include markdown formatting (no triple backticks).
Do not nest JSON objects.
Do not include arrays.
Return ONE and only ONE JSON object.
Do not include any other text or formatting.

It should look like this:
{{
    "Original": "This is a sentence and it can be simplified.",
    "Explanation": "Since the sentence contains two clauses, it can be simplified into two sentences.",
    "Simple": "This is a sentence. It can be simplified."
}}

Simplify the following sentence: {sentence}"""

# 3. Hallucination Checker Prompt
HALLUCINATION_CHECK_PROMPT_TEMPLATE_LOGIC = """You are evaluating sentence simplifications for language-to-logic translation.
    
Key Guidelines:
    - Splitting is the intended goal, so do not penalize for splitting sentences.
    - Do NOT penalize for removing conjunctions, discourse cues, or transitional phrases (e.g., "in addition to", "however") unless they signal necessary meaning.
    - Do NOT suggest adding back conjunctions or discourse markers unless their removal causes ambiguity or factual meaning loss.
    - Do NOT recommend merging unless the simplification causes ambiguity or removes essential logical relationships.
    - Each sentence should stand alone, with all subjects and actions clearly stated.
    - Pronouns are meant to be resolved; replacing them with specific nouns is preferred.
    - Do NOT penalize for loss of sentence “flow” or style; only assess whether logical content is preserved.
    - Ensure no hallucinations are present in the simplified version.
    - Identify factual meaning changes, such as dropped actions or incorrect subjects. Ignore superficial shifts in phrasing.
    - If all actions and agents are preserved and unambiguous, even if phrasing differs, consider it an acceptable simplification.
    - Provide a recommendation to fix the issues found, or "enthusiastically accept" if no issues are found.

Return a JSON object in the following format:
{{
    "original": "<original sentence>",
    "simplified": "<simplified sentence>",
    "extra_info": "<extra details found in the simplified sentence>",
    "missing_info": "<critical information missing from the simplified sentence>",
    "meaning_change": "<indicate if the meaning has changed in an unacceptable way>",
    "recommendation": "<recommendation to fix the issues found> or 'enthusiastically accept'>"
}}

Do not escape single quotes with a backslash.
Only use double quotes (") for keys and string values.
Do not include markdown formatting (no triple backticks).
Do not nest JSON objects.
Do not include arrays.

Here is the original and simplified sentence:
Original: "{original}"
Simplified: "{simplified}"
"""

# 4. Fix Hallucination Errors Prompt
FIX_HALLUCINATION_ERRORS_PROMPT_TEMPLATE_LOGIC = """Your task is to simplify a sentence for language to logic translation.

{context}

For the following sentence: "{sentence}"
The simplification "{simplified}" was detected to have the following issues: {issues_text}

Please correct the simplification accordingly and return a valid JSON object in the format:
{{
    "Original": <original sentence>,
    "Explanation": <explanation of simplification>,
    "Simple": <corrected simplified sentence>
}}
Simplify the following sentence: {sentence}"""

RETRY_ADD = """\nEnsure you return a single JSON object, nothing else. Here is an example of the correct format:
{{
    "original": "Jarrad ate the pizza and ate the cake.",
    "simplified": "Jarrad ate the pizza while sitting.",
    "extra_info": "Jarrad was not sitting in the original sentence.",
    "missing_info": "Jarrad eating cake is not mentioned in the simplified sentence, which is a critical detail.",
    "meaning_change": "The meaning has changed, since the simplified sentence implies Jarrad was sitting, which is not in the original.",
    "recommendation": "Correct the simplification to include the cake and remove the sitting detail. A possible simplification could be: 'Jarrad ate the pizza and cake.'"
}}

"""

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def parse_model_response(response_str):
    # Use regex to extract the JSON object from the response string
    #replace \' with ' to avoid issues with JSON parsing
    response_str = response_str.replace("\\'", "'")
    # replace triple backticks with nothing to avoid issues with JSON parsing
    response_str = response_str.replace("```", "")
    # replace \n with space to avoid issues with JSON parsing
    response_str = response_str.replace("\\n", ' ').strip()
    json_str = response_str.split('{')[-1] # extract the JSON part 
    json_str = '{' + json_str.split('}')[0] + '}' # add the opening and closing bracket
    try:
        parsed_json = json.loads(json_str)
        if 'Simple' in parsed_json or 'recommendation' in parsed_json:
            return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    print("Inside parse_model_response: Invalid JSON structure.")
    print(f"Response string: {response_str}")
    return None

# -----------------------------------------------------------------------------
# Main Functions
# -----------------------------------------------------------------------------

def simplify_sentence_adversarial(sentence, model=DEFAULT_MODEL, context_size=5, context_type='custom', complexity_filter=False, verbose=False):
    '''
    Adversarial sentence simplification that uses example contexts and iterative hallucination checking.
    
    Returns:
        tuple: (simplified_sentence, flag, retries) where flag is True if the simplification was accepted,
               or (original_sentence, False, retries) otherwise.
    '''

    issues_text = None
    if verbose:
        print(f'Simplifying sentence adversarially: {sentence}')
    
    if complexity_filter:
        complex_flag, _ = determine_complexity(sentence)
        if not complex_flag:
            return sentence, False, 0

    sentence_pairs = get_sentence_pairs(sentence, context_size, context_type) if context_size > 0 else None  
    context_examples = '\n'.join(f'Complex: {orig}\nSimple: {simp}' for orig, simp in sentence_pairs) if sentence_pairs else ''
    context = f'Here are some examples:\n{context_examples}\n' if context_examples else ''
    

    base_prompt = INITIAL_PROMPT_TEMPLATE_LOGIC.format(context=context, sentence=sentence)

    retries = 0
    temp = 0
    while retries < 100:
        if verbose:
            print(f'Attempt {retries+1}')
            print(f'For sentence: {sentence}')
        response = ollama.generate(model, prompt=base_prompt, options={'temperature': temp, 'num_predict': 1024})
        message = response['response'].replace('\n', ' ').strip()
        
        with open("ollama_log.txt", "a") as log:
            log.write(f"Attempt {retries+1}\n")
            log.write(f"Prompt: {base_prompt}\n")
            log.write(f"Response: {message}\n\n")
        
        parsed_json = parse_model_response(message)
        if parsed_json and "Simple" in parsed_json:
            simplified = parsed_json["Simple"]
            issues = ollama_hallucination_check(sentence, simplified, model=model)
            if not issues:
                #print(f"Enthusiastically accepted simplification: {simplified}")
                #print('resolving pronouns')
                simplified = resolve_pronouns(simplified)
                return simplified, True, retries
            else:
                new_issues_text = json.dumps(issues)
                if issues_text is not None and issues_text == new_issues_text:
                    # If the issues are the same as before, increase temperature and retry
                    temp += 0.1
                else:
                    issues_text = new_issues_text
                    temp = 0
                base_prompt = create_retry_prompt(context, sentence, simplified, issues_text)
                retries += 1
                temp += 0.1
                if temp > .5:
                    temp = 0
                if verbose:
                    print(f"Issues detected: {issues_text}. Retrying with modified prompt.")
                    print("New prompt:", base_prompt)
                time.sleep(3)
        else:
            print(f'No "Simple" key found in JSON response. Parsed JSON: {parsed_json}')
            base_prompt = IMPROPER_JSON_PROMPT_TEMPLATE_LOGIC.format(context=context, sentence=sentence)
            retries += 1
            temp += 0.1
            if temp > .5:
                temp = 0
            if verbose:
                print("Failed to parse JSON. Retrying with corrected prompt.")
                print("New prompt:", base_prompt)
            time.sleep(3)
    
    print(f'Warning: Did not receive a properly formatted and acceptable simplification after {retries} attempts. Returning original sentence.')
    return sentence, False, retries

def create_retry_prompt(context, sentence, simplified, issues_text):
    '''Creates a retry prompt using the fixed hallucination errors template'''
    prompt = FIX_HALLUCINATION_ERRORS_PROMPT_TEMPLATE_LOGIC.format(
        context=context, sentence=sentence, simplified=simplified, issues_text=issues_text)
    return prompt

def ollama_hallucination_check(original: str, simplified: str, model: str = DEFAULT_MODEL):
    '''
    Checks if the simplified sentence introduces hallucinations (extra details or missing key information)
    by comparing the original and simplified sentences.
    '''

    prompt = HALLUCINATION_CHECK_PROMPT_TEMPLATE_LOGIC.format(original=original, simplified=simplified)

    response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0, "num_predict": 1024})
    response_text = response["response"].strip()
    
    # Log the prompt and output
    with open("ollama_log.txt", "a") as log:
        log.write(f"Hallucination prompt: {prompt}\n")
        log.write(f"Hallucination output: {response_text}\n\n")
    
    issues = parse_model_response(response_text)
    while issues is None:
        print("Retrying hallucination check due to invalid JSON response.")
        response = ollama.generate(model=model, prompt=prompt + RETRY_ADD, options={"num_predict": 1024}) # retry with added instruction and default temperature
        response_text = response["response"].strip()
        with open("ollama_log.txt", "a") as log:
            log.write(f"Hallucination prompt: {prompt+RETRY_ADD}\n")
            log.write(f"Hallucination output: {response_text}\n\n")
        issues = parse_model_response(response_text)
    
    # Check if the recommendation is an enthusiastic accept.
    if "recommendation" in issues and "enthusiastically accept" in issues["recommendation"].lower():
        return {}
    
    # Remove keys with null values
    issues = {key: value for key, value in issues.items() if value is not None}
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

        simplified, simplified_flag, retries = simplify_sentence_adversarial(sentence, model=model_type)
        if not simplified_flag:
            print(f"Warning: Simplification failed for sentence: {sentence}. Returning original sentence.")
            simplified = sentence

        simplified_sentences.append(simplified)
    
    #join all sentences
    simplified_text = ' '.join(simplified_sentences)
    simplified_sentences = split_sentences(simplified_text)

    with open(output_file, 'w') as outfile:
        for sentence in simplified_sentences:
            outfile.write(sentence + '\n')
        
def resolve_pronouns(simplified):
    '''checks if simplified version contains pronouns and resolves them if it does'''           
    pronouns = False
    if check_pronouns_ollama(simplified, DEFAULT_MODEL):
        pronouns = True
    with open("ollama_log.txt", "a") as log:
        log.write(f"Sentence: {simplified}\nPronouns: {pronouns}\n\n")
    if pronouns:
        simplified = remove_pronouns(simplified, DEFAULT_MODEL)

    return simplified

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
            log.write(f"Pronoun resolution:\nOriginal: {sentence}\nResolved: {response_text}\n\n")

        return resolved_output
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None



def split_sentences(text):
    ''' Split a text into sentences using stanza. '''
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    doc = nlp(text)
    return [sentence.text for sentence in doc.sentences]




# -----------------------------------------------------------------------------
# Example Retrieval Functions
# -----------------------------------------------------------------------------

def get_custom_sentence_pairs(top_k=5):
    """Return the first top_k (original, simplified) pairs from CUSTOM_DIR."""
    orig_path = os.path.join(CUSTOM_DIR, 'custom.orig')
    simp_path = os.path.join(CUSTOM_DIR, 'custom.simp')

    with open(orig_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    with open(simp_path, 'r', encoding='utf-8') as f:
        simple_sentences = [line.strip() for line in f if line.strip()]

    if len(simple_sentences) != len(sentences):
        raise ValueError("Mismatch between custom.orig and custom.simp lengths")

    if top_k > len(sentences):
        top_k = len(sentences)

    return [(sentences[i], simple_sentences[i]) for i in range(top_k)]

def get_sentence_pairs(query, top_k=5, context_type='custom', return_indices=False):
    """
    For compatibility with existing callers.
    Only 'custom' is supported; ignores query and returns top_k custom pairs.
    """
    if top_k <= 0:
        return None
    if context_type != 'custom':
        raise ValueError('Only context_type="custom" is supported in the trimmed build.')
    # Indices mode not meaningful for custom-only; keep behavior simple.
    if return_indices:
        return list(range(top_k))
    return get_custom_sentence_pairs(top_k)
