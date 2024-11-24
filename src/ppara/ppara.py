import sys
import ollama
import os

PROMPT1 = 'rephrase the following sentence: '
PROMPT2 = 'Translate the sentence so that no metaphorical expressions are present. Make sure there is no figurative language, make the sentence as plain and literal as possible. MOST IMPORTANTLY, respond with ONLY the translated sentence.'
def call_ollama(prompt, model_type):
    """Call the Ollama model with the given prompt and model type."""
    try:
        # Use the ollama library to send the prompt to the model
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}])
        return(response["message"]["content"].replace('\n', ' ')) # Return the model response
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def process_file(input_file, output_file, model_type):
    """Process the input file, send prompts to the Ollama model, and write responses."""
    inProof = False
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Check if the line starts with '1 '
            if line.startswith('KBmanager.initializeOnce(): total init time in seconds'):
                inProof = True
            if inProof:
                print('recognized line')
                # Extract the part of the line after '1_'
                sentence = line.strip()
                if sentence:
                    prompt = PROMPT1 + sentence
                    # Call the Ollama model with the extracted prompt
                    response = call_ollama(prompt, model_type)
                    if response:
                        # Write the response to the output file
                        outfile.write(response + '\n')
            elif line.startswith('0'):
                outfile.write(line.strip()[2:] + '\n')
                
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_file> <model_type>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_type = sys.argv[3]

    process_file(input_file, output_file, model_type)
    print(f"Processing complete. Responses saved to {output_file}.")
