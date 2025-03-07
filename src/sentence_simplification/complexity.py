import ollama
import spacy
import textacy
import wordfreq

from wordfreq import zipf_frequency

nlp = spacy.load("en_core_web_sm")

def get_sentence_length(sentence):
    return len(sentence.split())

def get_average_frequency (sentence, language = 'en'):
    ''' Calculates complexity of a sentence based on the average frequency of usage in the language of words in the sentence.'''
    words = sentence.split()
    freq = 0
    for word in words:
        cumulative_freq += zipf_frequency(word, language)   # this number will be higher for more frequent words, and lower for less frequent words
    avg_freq = cumulative_freq/len(words)                       # average complexity of words in the sentence
    return avg_freq

def get_lexical_diversity(sentence):
    ''' Calculates lexical diversity of a sentence based on the number of unique words in the sentence.'''
    words = sentence.split()
    unique_words = set(words)
    return len(unique_words)/len(words) if words else 0

def get_readability(sentence):
    ''' Calculates readability of a sentence based on the number of words and average length of words in the sentence.'''
    words = sentence.split()
    word_length = sum(len(word) for word in words)
    return len(words)/word_length if word_length else 0


def get_complexity_dict(sentence):
    ''' Calculates metrics and returns a dictionary of complexity metrics of a sentence.'''
    return { 'sentence': sentence,
            'sentence_length': get_sentence_length(sentence), 
             'average_frequency': get_average_frequency(sentence),
             'lexical_diversity': get_lexical_diversity(sentence),
             'readability': get_readability(sentence) }


def check_pronouns_ollama(sentence, model_type):
    ''' Prompts the passed model_type to check if the sentence contains replacable pronouns. Returns true if it does, false if it does not.'''
    examples = """
    Here are some examples to help you understand pronoun resolution. A pronoun is considered resolvable if it can be replaced with a specific noun or proper noun in the sentence(s).

    1. 'Sarah loves painting. She spends hours on her artwork.' 
    Response: Since 'she' refers to Sarah, the answer is Yes.

    2. 'The dog barked at the cat. The cat ran away.' 
    Response: Since there are no pronouns in the sentence, the answer is No.

    3. 'John called Mike, but he didn’t answer.' 
    Response: Since 'he' refers to Mike, the answer is Yes.

    4. 'It is raining outside.' 
    Response: Since 'it' does not refer to anything specific, the answer is No.

    5. 'Alice met Bob. Alice gave Bob a book.' 
    Response: Since 'Alice' and 'Bob' are proper nouns, the answer is No.

    6. 'Tom saw Jerry. He waved at him.' 
    Response: Since 'he' refers to Tom, and 'him' refers to Jerry, the answer is Yes.

    7. 'Lisa found her keys on the table.' 
    Response: Since 'her' refers to Lisa, the answer is Yes.

    8. 'They say exercise is important.' 
    Response: Since 'they' does not refer to anything specific, the answer is No.
    """

    prompt = f"{examples}\nNow, does the following sentence(s) contain any pronouns that can be resolved / replaced? Answer 'Since <insert>, the answer is <Yes or No>.\nSentence(s): '{sentence}'.\nResponse:"

    try:
        # Use the ollama library to send the prompt to the model
        # temperature to 0 means there is no creativity, and responses are deterministic.
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
        if "the answer is no" in response["message"]["content"].lower():
            return False
        elif "the answer is yes" in response["message"]["content"].lower():
            return True
        else:
            print(f'WARNING: Invalid response from Ollama: {response["message"]["content"]} for sentence: {sentence}')
            return False
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None



def check_complexity_ollama(sentence, model_type):
    ''' Prompts the passed model_type to check if the sentence is common and easily understandable. Returns true if it is, false if it is not.'''


    prompt = "Does the following sentence have common structure and is easily understandable? Do not base your answer on the complexity of the subject or the nuace associated with the sentence. Base your answer solely on the structure and complexity of the language used. The sentence is: '" + sentence + "' Answer 'Yes' or 'No'."
    try:
        # Use the ollama library to send the prompt to the model
        # temperature to 0 means there is no creativity, and responses are determinic.sti
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
        if "yes" in response["message"]["content"].lower():
            return False
        elif "no" in response["message"]["content"].lower():
            return True
        else:
            raise Exception(f'Invalid response from Ollama: {response["message"]["content"]} for sentence: {sentence}')
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def check_complexity_hueristic(sentence, max_complexity_score = 30):
    '''Calculates if a sentence is overly complex, returns true if it is, false if it is not.'''
    # TODO : Implement a better way to calculate complexity score, right now it is just the length of the sentence
    complexity_score = get_sentence_length(sentence) 

    if complexity_score > max_complexity_score:
        return True
    else:
        return False



def get_max_depth(token, level=0):
    '''
    Calculate the maximum depth of a syntactic tree rooted at the given token.
    
    Args:
        token (spacy.tokens.Token): The root token of the syntactic tree.
        level (int): The current depth level (default is 0).
        
    Returns:
        list: A list of tuples containing token text and its depth.
    '''
    tree = [(token.text, level)]
    for child in token.children:
        tree.extend(get_max_depth(child, level + 1))
    return tree


def depth_tree_analysis(doc):
    '''
    Analyze the depth of the syntactic tree for each sentence in the given text.
    
    Args:
        doc (spacy.tokens.Doc): The input document to analyze.
        
    Returns:
        list: A list of tuples where each tuple contains:
              - Maximum depth of the syntactic tree for a sentence.
              - A list of tuples (token text, depth) for that sentence.
    '''
    results = []
    
    for sent in doc.sents:
        root = [token for token in sent if token.head == token][0]  # Get root token of sentence
        depth_tree = get_max_depth(root)
        max_depth = max(depth for _, depth in depth_tree)
        results.append(max_depth)
    
    return results


def get_sentence_length(doc):
    '''
    Calculate the length of the sentence in terms of the number of tokens.
    
    Args:
        doc (spacy.tokens.Doc): The input sentence to analyze.
        
    Returns:
        int: The number of tokens in the sentence.
    '''
    return len(doc)


def get_word_frequencies(doc):
    '''
    Calculate the zipf_frequency of each word in the given text.
    
    Args:
        doc (spacy.tokens.Doc): The input sentence to analyze.
        
    Returns:
        dict: A dictionary containing the word frequencies.
    '''

    return {token.text: wordfreq.zipf_frequency(token.text, 'en') for token in doc if token.pos_ != 'PUNCT'}



def clause_to_text(clause_tokens):
    '''
    Convert a list of tokens (forming a clause) back into a text string.
    '''
    # Sort tokens by their position in the original document
    sorted_tokens = sorted(clause_tokens, key=lambda t: t.i)
    return " ".join(t.text for t in sorted_tokens)

def split_clauses(doc):
    """
    Splits a spaCy doc into clauses based on the dependency tree.
    Handles "but" and "and" but NOT "or" to avoid loss of meaning.

    Args:
        doc (spacy.tokens.Doc): The parsed sentence.

    Returns:
        The list of clause texts.
    """
    clauses = []

    # Identify the main clause (using the ROOT token)
    roots = [token for token in doc if token.dep_ == "ROOT"]
    if not roots:
        return []
    
    main_root = roots[0]
    main_clause_tokens = list(main_root.subtree)
    clauses.append(main_clause_tokens)
    
    # Handle coordinated clauses ("and", "but") but NOT "or"
    for token in doc:
        if token.text.lower() in {"and", "but"} and token.dep_ == "cc":
            # Find the head of "and" or "but" and extract its subtree
            if token.head:
                clause_tokens = list(token.head.subtree)
                clauses.append(clause_tokens)

    # Convert token lists back into strings
    clause_texts = [clause_to_text(tokens) for tokens in clauses]
    return clause_texts


def determine_complexity(sentence, max_len_threshold=24, max_depth_threshold=4, min_freq_threshold=3):
    doc = nlp(sentence)
    frequencies = get_word_frequencies(doc)
    measured_min_freq = min(frequencies.values()) if frequencies else 0
    measured_max_depth, depth_tree = depth_tree_analysis(doc)
    sentence_length = get_sentence_length(doc)

    length_exceeded = sentence_length > max_len_threshold
    depth_exceeded = measured_max_depth > max_depth_threshold
    frequency_exceeded = measured_min_freq < min_freq_threshold

    is_complex = length_exceeded or depth_exceeded or frequency_exceeded

    complexity_dict = {
        'sentence_length': sentence_length,
        'max_depth': measured_max_depth,
        'min_freq': measured_min_freq,
        'length_exceeded': length_exceeded,
        'depth_exceeded': depth_exceeded,
        'frequency_exceeded': frequency_exceeded,
        'complex': is_complex
    }

    return is_complex, complexity_dict

if __name__ == '__main__':
    pass
