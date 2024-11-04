import os
import glob
import datetime
import ollama
import time
import random
from complexity import get_complexity_score


def simplify_file(file, date_time, max_allowed_complexity=30):
    '''
    
    '''
    with open(file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    output_lines = []

    model_list = ['llama3.2:1b', 'llama3.2:3b', 'llama3.1:8b', 'mistral']

    prompt_dict = {
                1 : "Your task is to simplify a long sentence into multiple, shorter sentences. Focus on splitting up ideas, ensuring that each idea is a separate sentence. Use clear and straightforward English. Replace pronouns with the appropriate noun. Return all sentences in one line, separated by periods. Keep only the essential words to preserve the sentence’s meaning. Most importantly, respond with only the sentences, nothing extra. The sentence is: '{sentence}'", 

                2 : "Simplify the following sentence by dividing it into smaller, simpler sentences. Ensure each distinct thought is expressed in its own sentence. Replace any pronouns with the corresponding noun, if you know it. Respond with all sentences in a single line, using periods to separate them. Avoid unnecessary words, keeping the main message intact. For example: 'As it was raining outside, Sarah sat by the fire, reflecting on the letter she had just received, while her brother paced the room, muttering about the strange visitor they had encountered earlier that day.' would become 'It was raining outside. Sarah sat by the fire. Sarah reflected on the letter she received. Sarah's brother paced the room. Sarah's brother talked about a visitor. Sarah and her brother encountered a visitor earlier in the day.' Most importantly, respond with only the sentences, nothing extra. The sentence is: '{sentence}'", 

                3 : "Your task is to simplify the sentence by converting it into multiple short sentences, ensuring clarity and precision. Break down each idea into separate sentences. Replace any pronouns with the correct noun. Return all the simplified sentences in one line, separated by periods. Strip out any non-essential words while maintaining the original meaning. Most importantly, respond with only the sentences, nothing extra. The sentence is: '{sentence}'",

                4 : "You are tasked with simplifying sentences into multiple, simpler sentences. If the sentence contains multiple ideas, split them into shorter sentences. Use common English words if possible. Respond with all sentences on one line, separated by periods. Remove words that are not essential in relaying the overall sentiment of the sentence. Replace pronouns with proper nouns, if possible. For example, 'John went to the store and he is sick' would be 'John went to the store. John is sick.' Most importantly, respond with only the sentences, nothing extra. The sentence is: '{sentence}'"}

    model = 'llama3.1:8b'
    prompt = prompt_dict[2]

    for line in lines:
        line = line.strip()
        if line:
            complexity_score = get_complexity_score(line)
            if complexity_score < max_allowed_complexity:    # If the sentence is already simple enough, leave it as it is
                output_lines.append(line)
            else:
                output_lines.append(simplify_sentence_llm(line, model, prompt))

    return output_lines
    


def simplify_sentence_llm(sentence, model, prompt):
    '''
    Simplify a sentence using the provided tokenizer and model.

    Args:
        sentence (str): The sentence to simplify.
        model (str): The model to use.
        prompt (str): The prompt to use.


    Returns:
        str: The simplified sentence.
    '''



    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt.format(sentence=sentence),
            },
        ],
        options={
            "temperature": 0.0,  # Makes responses more deterministic
            },

    )

    return(response["message"]["content"].replace('\n', ' '))

def end_program(start_time):
    #close file    
    print("Time taken in minutes:", (time.time() - start_time) / 60)
    print("Goodbye!")
    os._exit(0)




