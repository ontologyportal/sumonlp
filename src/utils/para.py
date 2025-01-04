import sys
import ollama
import os

PROMPT1 = 'Please restate the following quoted sentence in a more natural single English sentence."'


def call_ollama(prompt, model_type):
    """Call the Ollama model with the given prompt and model type."""
    try:
        # Use the ollama library to send the prompt to the model
        response = ollama.chat(model=model_type, messages=[{"role": "user", "content": prompt}])
        return(response["message"]["content"].replace('\n', ' ')) # Return the model response
    except Exception as e:
        print("error in call_ollama()")
        print("prompt was: ", prompt)
        print("model was: ", model_type)
        print(f"Error calling Ollama: {e}")
        return None

def process_file(input_file, output_file, model_type):
    """Process the input file, send prompts to the Ollama model, and write responses."""
    inProof = False
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if len(sentence.strip()) > 1:
                prompt = PROMPT1 + line + '"'
                #print("prompt: ", prompt)
                # Call the Ollama model with the extracted prompt
                response = call_ollama(prompt, model_type)
                if response:
                    #print("response: ", response)
                    # Write the response to the output file
                    outfile.write(response + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_file> <model_type>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_type = sys.argv[3]

    process_file(input_file, output_file, model_type)
    print(f"Processing complete. Responses saved to {output_file}.\n\n\n")
    process_entire_argument(input_file, "entire.txt", model_type)
    
