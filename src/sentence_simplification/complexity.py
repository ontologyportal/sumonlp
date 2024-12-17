from wordfreq import zipf_frequency   # requiremnet: pip install wordfreq

import ollama

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

    prompt = f"Does the following sentence contain **any pronouns** (like 'it', 'its', 'they', or 'them')?  '{sentence}'  Only answer 'Yes.' or 'No.' Do not treat repeated nouns as pronouns." 



    try:
        # Use the ollama library to send the prompt to the model
        # temperature to 0 means there is no creativity, and responses are deterministic.
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
        if "yes" in response["message"]["content"].lower():
            return True
        elif "no" in response["message"]["content"].lower():
            return False
        else:
            raise Exception(f'Invalid response from Ollama: {response["message"]["content"]} for sentence: {sentence}')
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None



def check_complexity_ollama(sentence, model_type):
    ''' Prompts the passed model_type to check if the sentence is common and easily understandable. Returns true if it is, false if it is not.'''


    prompt = "The following sentence is to be passed through a language to logic translator. The model is trained on a dataset of sentences that are common and easily understandable. The model will handle out of vocabulary words, like proper nouns, so you don't need to worry about that. Does the following sentence have common structure and is easily understandable? The sentence is: '" + sentence + "' Answer 'Yes' or 'No'."
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