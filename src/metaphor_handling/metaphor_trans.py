import sys
import ollama
import re

PROMPT1 = 'The following sentence contains metaphorical content:  '
PROMPT2 = 'Translate the sentence so that no metaphorical expressions are present. Make sure there is no figurative language, make the sentence as plain and literal as possible. MOST IMPORTANTLY, respond with ONLY the translated sentence.'
def call_ollama(prompt, model_type):
    """Call the Ollama model with the given prompt and model type."""
    try:
        # Use the ollama library to send the prompt to the model
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
        return(response["message"]["content"].replace('\n', ' ')) # Return the model response
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
            sentence = line.strip()[2:]  # Skip the initial '1 ' and get the rest
            if sentence:
                prompt = "Quick answer: Does the following sentence contain a blatant metaphor: " + sentence
                response = call_ollama(prompt, model_type)
                print (prompt)
                print (response)
                if "yes" in response.lower():
                    prompt = "Quick answer: Rephrase the following sentence with as few words as possible, without metaphorical content: " + sentence
                    response = call_ollama(prompt, model_type)
                    response = re.sub(r'\s*\([^)]*\)\s*$', '.', response) # Removes the parenthetical statement at the end of sentences.
                    outfile.write(response + '\n')
                else:
                    outfile.write(line.strip()[2:] + '\n')
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