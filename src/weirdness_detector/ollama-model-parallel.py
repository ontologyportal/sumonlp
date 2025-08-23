import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys
import os
import ollama

# Set up Ollama endpoint
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Adjust based on your setup
# MODEL_NAME = "custom-model-2-single"  # Replace with your specific model
# MODEL_NAME = "custom-model-2-single:latest"  # Replace with your specific model
# MODEL_NAME = "llama3.2:latest"  # Replace with your specific model
# MODEL_NAME = "llama3.2:3b-instruct-fp16"  # Replace with your specific model
# MODEL_NAME = "llama3.3"  # Replace with your specific model
# MODEL_NAME = "deepseek-r1:7b"  # Replace with your specific model
MODEL_NAME = "gpt-oss:latest"

MODE = ""


def evaluate_sentence(sentence):
    time.sleep(0.1)
    prompt = ""
    if (MODE == "STRICT"):
        #  Formulate the prompt (SCTRICT PROMPT) ~4-5% Sentences classified as Valid
        prompt = f"""
            Evaluate the following sentence for coherence and plausibility:

            Sentence: '{sentence}'

            Classify the sentence as 'Valid' if it makes sense, can logically appear in a book or newspaper, and is applicable to everyday tasks. Focus primarily on whether the object can logically be used with the given verb in a typical everyday situation without overcomplicating the analysis.

            Classify the sentence as 'Invalid' if it is illogical, self-contradictory, or impossible within commonly understood contexts.

            Return just one word 'Valid' or 'Invalid' with a brief explanation about your decision!
        """
    elif (MODE == "RELAXED"):
        # Relaxed Prompt
        prompt = f"""

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

            Return just one word 'Valid' or 'Invalid' nothing else!

            Sentence: '{sentence}'
        """
    else:
        print("INVALID MODE!")

    print("Prompt: " + prompt)
    # Send the request to Ollama
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 5,
            "stream": True,
            "options": {
                "temperature": 0
            }
        },
        stream = True
    )

    print("RESPONSE: " + response.status_code)
    if response.status_code == 200:
        full_response = ""
        try:
            # Process each line of the streaming response
            for line in response.iter_lines():
                if line:
                    print("RESPONSE: " + line)
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


def process_sentences(sentences):
    valid_sentences = []
    invalid_sentences = []

    # Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        #results = list(executor.map(evaluate_sentence, sentences))
        results = list(tqdm(executor.map(evaluate_sentence, sentences), total=len(sentences), desc="Processing sentences"))

    for sentence, result in zip(sentences, results):
        print(f"Sentence: {sentence}\nResult: {result}\n")
        if "Valid" in result:
            valid_sentences.append(sentence)
        elif "Invalid" in result:
            invalid_sentences.append(sentence)

    return valid_sentences, invalid_sentences


def run_detector(mode_p, file_path, file_output):
    global MODE
    MODE = mode_p
    print("-"*50)
    print(f"Weirdness Detector Started")
    print("Using " + MODE + " Model")
    print("-"*50)

    # Read sentences from file
    with open(file_path, "r") as file:
        sentences_to_test = file.read().splitlines()

    # Process sentences in parallel
    valid_sentences, invalid_sentences = process_sentences(sentences_to_test)

    # Save results to file
    with open(file_output, "w") as f:
        f.write(f"Total sentences processed successfully: {len(valid_sentences) + len(invalid_sentences)}\n")
        f.write(f"Valid sentences: {len(valid_sentences)} \n")
        f.write(f"Invalid sentences: {len(invalid_sentences)} \n\n")

        f.write("---- VALID SENTENCES ----\n")
        f.writelines([f"{sentence}\n" for sentence in valid_sentences])
        f.write("\n---- INVALID SENTENCES ----\n")
        f.writelines([f"{sentence}\n" for sentence in invalid_sentences])

    print(f"Valid sentences: {len(valid_sentences)}")
    print(f"Invalid sentences: {len(invalid_sentences)}")

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        filename = os.path.basename(file_path)
        filename_only = os.path.splitext(filename)[0]
        directory = os.path.dirname(file_path)
    else:
        print("No filepaths provided")

    if len(sys.argv) > 2:
        MODEL_NAME = sys.argv[2]
    start_time = time.time()

    run_detector("RELAXED", file_path, os.path.join(directory+'results', filename_only+'_weird_det_results_RELAXED.txt'))
    run_detector("STRICT", file_path, os.path.join(directory+'results', filename_only+'_weird_det_results_STRICT.txt'))

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the processing time
    print(f"Processing completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()