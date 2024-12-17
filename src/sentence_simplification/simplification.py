import os
import glob
import datetime
import ollama
import time
import random
from complexity import *
import stanza

def call_ollama(prompt, model_type):
    """Call the Ollama model with the given prompt and model type."""
    try:
        # Use the ollama library to send the prompt to the model
        # temperature to 0 means there is no creativity, and responses are deterministic.
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
        with open("ollama_log.txt", "a") as log:
            log.write(f"Prompt: {prompt}\nResponse: {response['message']['content']}\n\n")
        return(response["message"]["content"].replace('\n', ' ')).replace("Here's a rephrased version:  ", "") # Return the model response
    
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def simplify_file(input_file, output_file, model_type):
    ''' Simplify a file using the provided tokenizer and model. Flows through the following steps:
        1. check if sentence is complex and simplifies it if it is.
        2. checks if sentence contains pronouns and replaces them with accurate nouns if possible.
        3. joins all sentences and uses stanza to separate sentences to write onto individual lines of output. '''

    simplified_sentences = []  

    for sentence in open(input_file, 'r'):
        sentence = sentence.strip()
        if check_complexity_ollama(sentence, model_type) or check_complexity_hueristic(sentence):  
            sentence = simplify_sentence_llm(sentence, model_type)
        if check_pronouns_ollama(sentence, model_type):
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

    prompt = "Replace all pronouns with the appropriate nouns in the following sentences. If the pronoun cannot be resolved leave it alone. IMPORTANTLY, respond with just the sentencess, no additional commentary. The sentencess are: '" + sentence + "'"

    response = call_ollama(prompt, model)
    return response


def simplify_sentence_llm(sentence, model):
    '''Simplifies a sentence by concatenating prompt to it and using the passed model. '''

    prompt = "**IMPORTANT**: Output without additional commentary! Just the answer: Please split the sentence into several small sentences in a way that the vocabulary, grammar and sentence structure are common and easily understandable. Do not use pronouns unless absolutely necessary. The sentence is: '" + sentence + "'"

    response = call_ollama(prompt, model)
    print(f'Original: {sentence}\nSimplified: {response}')
    return response

def split_sentences(text):
    ''' Split a text into sentences using stanza. '''
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    doc = nlp(text)
    return [sentence.text for sentence in doc.sentences]

def end_program(start_time):
    #close file    
    print("Time taken in minutes:", (time.time() - start_time) / 60)
    print("Goodbye!")
    os._exit(0)




