import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys
import os
import ollama
import re

# Set up Ollama endpoint
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Adjust based on your setup
# MODEL_NAME = "custom-model-2-single"  # Replace with your specific model
# MODEL_NAME = "custom-model-2-single:latest"  # Replace with your specific model
MODEL_NAME = "llama3.2:latest"  # Replace with your specific model
# MODEL_NAME = "llama3.2:3b-instruct-fp16"  # Replace with your specific model
# MODEL_NAME = "llama3.3"  # Replace with your specific model
# MODEL_NAME = "deepseek-r1:7b"  # Replace with your specific model
MODEL_NAME = "gpt-oss:latest"

MODE = ""

import json


def ensure_model_downloaded(model_name: str) -> bool:
    models_info = ollama.list()
    models = models_info.get('models', [])

    # Determine if each item is dict or string
    if models and isinstance(models[0], dict):
        # Try different keys that might hold the model name
        model_names = [m.get('model') or m.get('name') for m in models]
    else:
        model_names = models  # assume list of strings

    full_model_name = f"{model_name}"
    if full_model_name in model_names:
        print(f"Model '{full_model_name}' already downloaded.")
        return True

    print(f"Model '{full_model_name}' not found locally, pulling it now...")
    try:
        for status in ollama.pull(model_name, stream=True):
            percent = status.get("progressPercent", 0)  # adapt field name as needed
            print(f"Downloading model: {percent:.2f}%", end="\r", flush=True)
        print(f"Model '{full_model_name}' successfully downloaded.")
        return True
    except Exception as e:
        print(f"Failed to download model '{full_model_name}': {e}")
        return False



def extract_classification(response):
    if isinstance(response, str):
        # Use regex to find the first {...} JSON block
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response.")
        try:
            response = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}")

    # Ensure classification field exists
    if "classification" not in response:
        raise KeyError("'classification' field not found in response.")

    return response["classification"]



def evaluate_sentence(sentence):
    try:
        prompt = ""
        if (MODE == "STRICT"):
            #  Formulate the prompt (SCTRICT PROMPT) ~4-5% Sentences classified as Valid
            prompt = r"""
                You are a text classification assistant.  
                Your task is to classify a user-provided sentence as either "Valid" or "Invalid". 
                
                Rules:  
                - Always respond in valid JSON format only.  
                - Do not include any text outside of the JSON response.  
                - The JSON must follow this schema exactly:  
                
                {
                  "input_sentence": "<the sentence being classified>",
                  "classification": "<valid | invalid>",
                  "confidence": <number between 0.0 and 1.0>,
                  "explanation": "<short reason for the classification>"
                }
                
                - Ensure the JSON is syntactically valid (parsable).  
                - Do not include formatting instructions, markdown, or extra commentary in the response.
                
                Evaluate the following sentence for coherence and plausibility:

                Sentence: '{sentence}'

                Classify the sentence as 'Valid' if it makes sense, can logically appear in a book or newspaper, and is applicable to everyday tasks. Focus primarily on whether the object can logically be used with the given verb in a typical everyday situation without overcomplicating the analysis.

                Classify the sentence as 'Invalid' if it is illogical, self-contradictory, or impossible within commonly understood contexts.
            """
        elif (MODE == "RELAXED"):
            # Relaxed Prompt
            prompt = r"""
                You are a text classification assistant.  
                Your task is to classify a user-provided sentence as either "Valid" or "Invalid". 
                
                Rules:  
                - Always respond in valid JSON format only.  
                - Do not include any text outside of the JSON response.  
                - The JSON must follow this schema exactly:  
                
                {
                  "input_sentence": "<the sentence being classified>",
                  "classification": "<valid | invalid>",
                  "confidence": <number between 0.0 and 1.0>,
                  "explanation": "<short reason for the classification>"
                }
                
                - Ensure the JSON is syntactically valid (parsable).  
                - Do not include formatting instructions, markdown, or extra commentary in the response.
                Classify a sentence as Invalid only if it describes a scenario that is completely beyond any conceivable reality, even under the most imaginative or hypothetical conditions. This includes cases where:
                The action described is fundamentally impossible under any realistic or fictional context.
                The sentence contains elements that contradict basic universal concepts (e.g., logical impossibilities, contradictions with common human experience).
                In all other cases, classify the sentence as Valid, allowing for unusual, rare, or imaginative scenarios that could happen under specific or extraordinary circumstances. If a human can conceive of the event happening in some form—no matter how unlikely—it should be considered valid.

                Examples:

                The monk borrowed a lion from Cheyenne. → Valid, since borrowing exotic animals, though rare, is possible.

                Victoria was cleaning a locomotive. → Valid, since this is a common, realistic task.

                Duncan doesn't say that "The anthropologist won't be smoking a motion picture." → Valid, because people can express anything, even nonsense.

                Constance will be dreaming of a paper. → Valid, since dreams can contain anything imaginable.

                Gary farms a human corpse. → Valid, since farming techniques could be metaphorically or ethically debated but not physically impossible.

                The refugee will suffer a vehicle brake. → Valid, as metaphorical interpretations could apply in an abstract sense.

                The mountain danced with joy. → Invalid, as inanimate objects do not possess emotions or mobility in any conceivable context.

                Time traveled back into itself to rewrite history. → Invalid, as it contradicts fundamental concepts of causality.

                Sentence: '{sentence}'
            """
        else:
            print("INVALID MODE!")

        # Generate response from the model
        returnObj = ollama.generate(model=MODEL_NAME, prompt=prompt)
        print(returnObj['response'])
        return extract_classification(returnObj['response'])

    except Exception as e:
        print("Error sending prompt to Ollama:", e)
        return None


def process_sentences(sentences):
    valid_sentences = []
    invalid_sentences = []
    unclassified_sentences = []

    # Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        #results = list(executor.map(evaluate_sentence, sentences))
        results = list(tqdm(executor.map(evaluate_sentence, sentences), total=len(sentences), desc="Processing sentences"))

    for sentence, result in zip(sentences, results):
        print(f"Sentence: {sentence}\nResult: {result}\n")
        if "invalid" in result.lower():
            invalid_sentences.append(sentence)
        elif "valid" in result.lower():
            valid_sentences.append(sentence)
        else:
            print("COULD NOT CLASSIFY this sentence: " + result)
            unclassified_sentences.append(sentence)

    return valid_sentences, invalid_sentences, unclassified_sentences


def run_detector(mode_p, file_path, file_output):
    global MODE
    MODE = mode_p
    print("-"*50)
    print(f"Weirdness Detector Started")
    print("Using " + MODE + " Mode and " + MODEL_NAME + " model" )
    print("-"*50)
    print("-"*50)

    # Read sentences from file
    with open(file_path, "r") as file:
        sentences_to_test = file.read().splitlines()

    # Process sentences in parallel
    valid_sentences, invalid_sentences, unclassified_sentences = process_sentences(sentences_to_test)

    # Save results to file
    with open(file_output, "w") as f:
        f.write(f"Total sentences processed successfully: {len(valid_sentences) + len(invalid_sentences)}\n")
        f.write(f"Valid sentences: {len(valid_sentences)} \n")
        f.write(f"Invalid sentences: {len(invalid_sentences)} \n\n")

        f.write("---- VALID SENTENCES ----\n")
        f.writelines([f"{sentence}\n" for sentence in valid_sentences])
        f.write("\n---- INVALID SENTENCES ----\n")
        f.writelines([f"{sentence}\n" for sentence in invalid_sentences])
        f.write("\n---- Unclassified SENTENCES ----\n")
        f.writelines([f"{sentence}\n" for sentence in unclassified_sentences])

    print(f"Valid sentences: {len(valid_sentences)}")
    print(f"Invalid sentences: {len(invalid_sentences)}")
    print(f"Unclassified sentences: {len(unclassified_sentences)}")

def main():
    global MODEL_NAME
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        filename = os.path.basename(file_path)
        filename_only = os.path.splitext(filename)[0]
        directory = os.path.dirname(file_path)
    else:
        print("No filepaths provided")

    if len(sys.argv) > 2:
        MODEL_NAME = sys.argv[2]
    if not ensure_model_downloaded(MODEL_NAME):
        sys.exit(0)

    start_time = time.time()

    run_detector("RELAXED", file_path, os.path.join(directory+'results', filename_only+'_weird_det_results_RELAXED.txt'))
    run_detector("STRICT", file_path, os.path.join(directory+'results', filename_only+'_weird_det_results_STRICT.txt'))

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the processing time
    print(f"Processing completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()