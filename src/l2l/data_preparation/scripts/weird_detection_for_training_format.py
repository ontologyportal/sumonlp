import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
import os

# Set up Ollama endpoint
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Adjust based on your setup
# MODEL_NAME = "custom-model-2-single"  # Replace with your specific model
# MODEL_NAME = "custom-model-2-single:latest"  # Replace with your specific model
# MODEL_NAME = "llama3.2:latest"  # Replace with your specific model
MODEL_NAME = "llama3.2:3b-instruct-fp16"  # Replace with your specific model
# MODEL_NAME = "llama3.3"  # Replace with your specific model
# MODEL_NAME = "deepseek-r1:7b"  # Replace with your specific model






def evaluate_sentence(json_object):

    time.sleep(0.1)

    sentence = json_object["input"]

    #  Formulate the prompt (SCTRICT PROMPT) ~4-5% Sentences classified as Valid
    prompt = f"""
      Evaluate the following sentence for coherence and plausibility:

      Sentence: '{sentence}'

      Classify the sentence as 'Valid' if it makes sense, can logically appear in a book or newspaper, and is applicable to everyday tasks. Focus primarily on whether the object can logically be used with the given verb in a typical everyday situation without overcomplicating the analysis.

      Classify the sentence as 'Invalid' if it is illogical, self-contradictory, or impossible within commonly understood contexts.

      Return just one word 'Valid' or 'Invalid' with a brief explanation about your decision!
    """

    # Send the request to Ollama
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 2,
            "options": {
              "num_predict":2,
              "temperature": 0
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
            return full_response
        except json.JSONDecodeError:
            return "Error: Unable to parse JSON in streaming response."
    else:
        return f"Error: {response.status_code} - {response.text}"


def process_sentences(json_objects):
    valid_sentences = []
    invalid_sentences = []

    # Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_sentence, json_objects))

    for json_object, result in zip(json_objects, results):
        # print(f"Sentence: {sentence}\nResult: {result}\n")
        if "Valid" in result:
            valid_sentences.append(json_object["input"])
        elif "Invalid" in result:
            invalid_sentences.append(json_object["input"])

    return valid_sentences, invalid_sentences


start_time = time.time()

sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l')
# File paths
file_path = os.path.join(l2l_home, 'data_preparation', 'data', 'input_l2l.json')
file_output = os.path.splitext(file_path)[0]+"_valid_sentences.txt"


print("-"*50)
print(f"Weirdness Detector Started")
print(f"Stricted Model")
print("-"*50)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process sentences in parallel
valid_sentences, invalid_sentences = process_sentences(data)

# Save results to file
with open(file_output, "w") as f:
    f.write(f"Valid sentences: {len(valid_sentences)} \n")
    f.write(f"Invalid sentences: {len(invalid_sentences)} \n\n")
    f.write("---- VALID SENTENCES ----\n")
    f.writelines([f"{sentence}\n" for sentence in valid_sentences])
    # f.write("\n---- INVALID SENTENCES ----\n")
    # f.writelines([f"{sentence}\n" for sentence in invalid_sentences])

print(f"Valid sentences: {len(valid_sentences)}")
print(f"Invalid sentences: {len(invalid_sentences)}")

end_time = time.time()
elapsed_time = end_time - start_time

# Print the processing time
print(f"Processing completed in {elapsed_time:.2f} seconds.")
