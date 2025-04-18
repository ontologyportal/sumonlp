import sys
import ollama
import os

PROMPT1 = 'Please restate the following quoted sentence in a more natural single English sentence."'
PROMPT2 = ' Extremely import rules to follow: Respond with just the sentences. Do not add apostrophes around the sentences. Do not add additional commentary. DO NOT ADD ADDITIONAL INFORMATION. The sentence to simplify is: "'


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
        while True:
            form = infile.readline()
            line = infile.readline()
            if not form or not line:
                break
            if len(line.strip()) > 1:
                prompt = PROMPT1 + PROMPT2 + line + '"'
                print("prompt: ", prompt)
                # Call the Ollama model with the extracted prompt
                response = call_ollama(prompt, model_type)
                if response:
                    outfile.write(form)
                    outfile.write(line)
                    print("response: ", response)
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

    
