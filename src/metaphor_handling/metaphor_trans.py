import sys
import re
import ollama

class MetaphorTranslator:

    def __init__(self, model_type: str):
        # Initialize model type
        self.model_type = model_type



    def call_ollama(self, prompt: str):
        try:
            response = ollama.chat(model=self.model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
            return response["message"]["content"].replace('\n', ' ')
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None

    def trans_metaphor(self, line: str):
        if line.startswith('0'):
            sentence = line.strip()[2:]
            if sentence:
                prompt = f"Quick answer: Does the following sentence contain a blatant metaphor: {sentence}"
                response = self.call_ollama(prompt)
                if "yes" in response.lower():
                    prompt = f"Quick answer: Rephrase the following sentence with as few words as possible, without metaphorical content: {sentence}"
                    response = self.call_ollama(prompt)
                    response = re.sub(r'\s*\([^)]*\)\s*$', '.', response)  # Removes trailing parentheticals
                    return response
                else:
                    return line.strip()[2:]
            else:
                return sentence
        else:
            prompt = f"Quick answer: Rephrase the following sentence with as few words as possible, without metaphorical content: {line.strip()[2:]}"
            response = self.call_ollama(prompt)
            response = re.sub(r'\s*\([^)]*\)\s*$', '.', response)  # Removes trailing parentheticals
            return response

    def process_file(self, input_file: str, output_file: str):
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                line = line.strip()
                result = self.trans_metaphor(line)
                outfile.write(result + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_file> <model_type>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_type = sys.argv[3]

    translator = MetaphorTranslator(model_type=model_type)
    translator.process_file(input_file, output_file)
    print(f"Processing complete. Responses saved to {output_file}.")
