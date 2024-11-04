# Readability
from wordfreq import zipf_frequency   # requiremnet: pip install wordfreq

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


def get_complexity_score(sentence):
    ''' Calculates complexity score of a sentence based on the complexity metrics.'''
    # TODO : Implement a better way to calculate complexity score, right now it is just the length of the sentence
    return get_sentence_length(sentence)