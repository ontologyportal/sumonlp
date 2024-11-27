import sys
import ollama
import os
import re

def call_ollama(prompt, model_type):
    """Call the Ollama model with the given prompt and model type."""
    try:
        # Use the ollama library to send the prompt to the model
        # temperature to 0 means there is no creativity, and responses are deterministic.
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
        return(response["message"]["content"].replace('\n', ' ')).replace("Here's a rephrased version:  ", "") # Return the model response
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def process_file(input_file, output_file, model_type):
    """Process the input file, send prompts to the Ollama model, and write responses."""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Check if the line starts with '1 '
            #if line.startswith('1'):


            #print('recognized line')
            # Extract the part of the line after '1_'
            sentence = line.strip()
            if sentence:
                # Vocabulary change.

                prompt = "Just the answer: Are the vocabulary, grammar and sentence structure in this sentence common and easily understandable: " + sentence
                response = call_ollama(prompt, model_type)
                if "no" in response.lower():
                    prompt = "**IMPORTANT**: Output without additional commentary! Just the answer: Please split the sentence into several small sentences. In a way that the vocabulary, grammar and sentence structure are common and easily understandable.: " + sentence
                    response = call_ollama(prompt, model_type)
                    response = re.sub(r'\s*\([^)]*\)\s*$', '.', response) # Removes the parenthetical statement at the end of sentences.
                    prompt = "**IMPORTANT**: Output without additional commentary! Just the answer: Does the following contain pronouns 'its', 'it', 'they', or 'them': " + response
                    contains_pronouns = call_ollama(prompt, model_type)
                    print("Before pronoun replacement: " + response)
                    print(contains_pronouns)
                    if "yes" in contains_pronouns.lower():
                        prompt = "**IMPORTANT**: Output without additional commentary! Just the answer: Replace all 'its', 'it', 'they', or 'them' with accurate nouns in the following sentences: " + response
                        response = call_ollama(prompt, model_type)
                    outfile.write(response + '\n')
                else:
                    outfile.write(sentence + '\n')
            #if sentence:
            #    prompt = PROMPT1 + sentence + PROMPT2
                # Call the Ollama model with the extracted prompt
            #    response = call_ollama(prompt, model_type)
            #    if response:
                    # Write the response to the output file
            #        outfile.write(response + '\n')
            #elif line.startswith('0'):
            #    outfile.write(line.strip()[2:] + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_file> <model_type>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_type = sys.argv[3]

    process_file(input_file, output_file, model_type)
    print(f"Processing complete. Responses saved to {output_file}.")