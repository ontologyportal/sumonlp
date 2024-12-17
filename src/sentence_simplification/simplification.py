import os
import glob
import datetime
import ollama
import time
import random
from complexity import *
import stanza
import tqdm
from util import *

def call_ollama(prompt, model_type):
    """Call the Ollama model with the given prompt and model type."""
    try:
        # Use the ollama library to send the prompt to the model
        # temperature to 0 means there is no creativity, and responses are deterministic.
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
        response = response["message"]["content"]
        response = remove_numbering(response)
        response = response.replace('\n', ' ')
        with open("ollama_log.txt", "a") as log:
            log.write(f"Prompt: {prompt}\nResponse: {response}\n\n")
        return(response)
    
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def simplify_file(input_file, output_file, model_type):
    ''' Simplify a file using the provided tokenizer and model. Flows through the following steps:
        1. check if sentence is complex and simplifies it if it is.
        2. checks if sentence contains pronouns and replaces them with accurate nouns if possible.
        3. joins all sentences and uses stanza to separate sentences to write onto individual lines of output. '''

    simplified_sentences = []  

    for sentence in tqdm.tqdm(open(input_file, 'r').readlines()):
        sentence = sentence.strip()

        complex = False
        if check_complexity_ollama(sentence, model_type) or check_complexity_hueristic(sentence):  
            complex = True
        with open("ollama_log.txt", "a") as log:
            log.write(f"Sentence: {sentence}\nComplex: {complex}\n\n")

        if complex:
            sentence = simplify_sentence_llm(sentence, model_type)

        pronouns = False
        if check_pronouns_ollama(sentence, model_type):
            pronouns = True
        with open("ollama_log.txt", "a") as log:
            log.write(f"Sentence: {sentence}\nPronouns: {pronouns}\n\n")
        if pronouns:
            sentence = remove_pronouns(sentence, model_type)

        simplified_sentences.append(sentence)
    
    #join all sentences
    simplified_text = ' '.join(simplified_sentences)
    simplified_sentences = split_sentences(simplified_text)

    with open(output_file, 'w') as outfile:
        for sentence in simplified_sentences:
            outfile.write(sentence + '\n')
        
            

def remove_pronouns(sentence, model):
    '''
    Remove pronouns from a sentence passing prompt and sentence through passed model. 
    '''

    prompt = prompt = "Perform coreference resolution on the following sentence. Follow these rules EXACTLY: Return ONLY the resolved sentence. Do not number the sentences. Do not add quotes, apostrophes, or any additional punctuation around the sentence. Do not include explanations, commentary, or any other text. If a pronoun cannot be resolved, leave it unchanged. Sentence: '" + sentence + "'"


    response = call_ollama(prompt, model)
    return response


def simplify_sentence_llm(sentence, model):
    '''Simplifies a sentence by concatenating prompt to it and using the passed model. '''

    prompt = "Split the following sentence into several small sentences in a way that the grammar and sentence structure are common and easily understandable. Extremely import rules to follow: Respond with just the sentences. Do not add apostrophes around the sentences. Do not add additional commentary. DO NOT ADD ADDITIONAL INFORMATION. The sentence to simplify is: '" + sentence + "'"

    response = call_ollama(prompt, model)
    return response

def split_sentences(text):
    ''' Split a text into sentences using stanza. '''
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    doc = nlp(text)
    return [sentence.text for sentence in doc.sentences]




