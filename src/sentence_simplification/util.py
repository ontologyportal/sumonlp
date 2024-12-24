import os
import time
import datetime
import sys
import stanza
import re

def cleanup_input(input_file):
    ''' Clean up input file by removing blank lines and verifying every line contains just one sentence '''
    input_file = remove_blank_lines(input_file)   # Remove blank lines and whitespace

    sentences = []
    with open(input_file, 'r') as file:
        text = file.read()
    
    nlp = stanza.Pipeline('en', processors='tokenize')
    for line in text.split('\n'):
        sentences.extend(split_sentences(line, nlp))
    
    with open(input_file, 'w') as file:
        for sentence in sentences:
            file.write(sentence + '\n')

    return input_file

def remove_blank_lines(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    with open(input_file, 'w') as file:
        for line in lines:
            if line.strip():
                file.write(line)
    return input_file

def split_sentences(text, nlp):
    ''' Split text into sentences using stanza '''
    doc = nlp(text)
    sentences = []
    for sentence in doc.sentences:
        sentence = ' '.join([word.text for word in sentence.words])
        sentences.append(sentence)
    return sentences


def end_program(start_time):
    #close file    
    print("Time taken in minutes:", (time.time() - start_time) / 60)
    print("Goodbye!")
    os._exit(0)

def remove_numbering(response):
    # Remove leading numbers, optional dots, and spaces
    cleaned_response = re.sub(r'^\d+\.\s*', '', response, flags=re.MULTILINE)
    return cleaned_response