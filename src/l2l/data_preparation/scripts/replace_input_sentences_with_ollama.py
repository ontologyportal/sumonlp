import requests
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor

"""
Input: JSON training data.
Output: JSON training data.

Reword the input sentences with Ollama.
"""

# Set up Ollama endpoint
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Adjust based on your setup
# MODEL_NAME = "custom-model-2-single"  # Replace with your specific model
# MODEL_NAME = "custom-model-2-single:latest"  # Replace with your specific model
# MODEL_NAME = "llama3.2:latest"  # Replace with your specific model
MODEL_NAME = "llama3.2:3b-instruct-fp16"  # Replace with your specific model
# MODEL_NAME = "llama3.3"  # Replace with your specific model

def evaluate_sentence(json_object):
    time.sleep(0.1)

    sentence = json_object["input"]

    prompt = f"""
      You are an advanced language model that specializes in rewording sentences while preserving their original meaning.
      Your task is to rewrite each sentence using different words or structure, but the meaning must remain exactly the same.

      Rules:

      Keep the logical and contextual meaning unchanged.
      Do not omit or add any new information.
      Use synonyms, different sentence structures, or rewording to achieve the goal.
      Keep the sentence grammatically correct and natural.
      Return only the new sentence, nothing else!

      Example:
      Input: "Ignatius watches whether to subject."
      Output: "Ignatius observes whether to comply."

      Input: "The anthropologist wasn't inheriting a class from Carl."
      Output: "The anthropologist wasn't deriving a class from Carl."

      Now, reword the following sentence without changing its meaning:

      Sentence: '{sentence}'

    """


    # Send the request to Ollama
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 500,
            "options": {
              "num_predict":500,
              "temperature": 0.3
            }
        }
    )

    if response.status_code == 200:
        full_response = ""
        try:
            # Process each line of the streaming response
            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line)
                    # Append the response text
                    full_response += line_data.get("response", "")
                    # Break if the response is marked as complete
                    if line_data.get("done", False):
                        break
            result = {
                "input": full_response.strip(),
                "output": json_object["output"]
            }
            return result
        except json.JSONDecodeError:
            return "Error: Unable to parse JSON in streaming response."
    else:
        return f"Error: {response.status_code} - {response.text}"


def process_sentences(sentences):

    # Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_sentence, sentences))

    for sentence, result in zip(sentences, results):
        old = sentence["input"]
        new = result["input"]
        print(f"Old Sentence: {old}\nNew Sentence: {new}\n")

    return results


start_time = time.time()

sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l')
# File paths
file_path = os.path.join(l2l_home, 'data_preparation', 'data', 'input_l2l.json')

file_output = os.path.splitext(file_path)[0]+"_ollama_reword.json"

print("-"*50)
print(f"Input Sentence Replacement Started")
print("-"*50)

# Load JSON data
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process sentences in parallel
results = process_sentences(data)

# Save results to file
with open(file_output, "w") as f:
    json.dump(results, f, indent=4)

print(f"Number of Sentences before: {len(data)}")
print(f"Number of Sentences after : {len(results)}")

end_time = time.time()
elapsed_time = end_time - start_time

# Print the processing time
print(f"Processing completed in {elapsed_time:.2f} seconds.")
