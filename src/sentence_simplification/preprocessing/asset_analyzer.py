import os
import spacy
import datetime
import csv
import numpy as np
import random
import sys
from scipy.stats import linregress
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from complexity import get_max_depth, depth_tree_analysis

ASSET_DIR = '../data/corpora/asset-main/dataset'
EMBEDDINGS_DIR = '../data/embeddings'
RESULTS_DIR = '../data/results'

nlp = spacy.load("en_core_web_sm")   # maybe switch to lg model at production

def count_sentences(line):
    doc = nlp(line)
    return len(list(doc.sents))

def average_sent_per_turk():
    results = {}
    for i in range(10):   # for simp.0 to simp.9
        filename = f'{ASSET_DIR}/asset.valid.simp.{i}'
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
                total_sentences = sum(count_sentences(line) for line in lines)
                average_sentences_per_line = total_sentences / len(lines) if lines else 0
                results[f'simp.{i}'] = average_sentences_per_line
        except FileNotFoundError:
            print(f"Warning: File {filename} does not exist!")

    highest_split = max(results, key=results.get)
    print("Average sentences per line for each simplification set:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")

def get_sentence_pairs(dir, set = 'valid'):
    '''
    Get sentence pairs from the test set. For each line in the complex file,
    collect the corresponding lines from simp.0 to simp.9 files.
    '''
    # read complex sentences
    with open(f'{dir}/asset.{set}.orig', 'r') as f:
        complex_sentences = f.readlines()

    for i, sentence in enumerate(complex_sentences):   # strip the newline characters
        complex_sentences[i] = sentence.strip()


    # list to hold all simplifications
    simple_sentences = [[] for _ in range(len(complex_sentences))]

    for filenum in range(10):  # For simp.0 to simp.9
        simp_file_path = f'{dir}/asset.{set}.simp.{filenum}'
        if os.path.exists(simp_file_path): 
            with open(simp_file_path, 'r') as simp_file:
                for i, line in enumerate(simp_file):
                    simple_sentences[i].append(line.strip())  # Add the line to the corresponding index
        else:
            print(f"Warning: File {simp_file_path} does not exist!")

    return complex_sentences, simple_sentences

def get_sentence_length(sentence):
    doc = nlp(sentence)
    return len(doc)

def get_num_sentences(simplified_versions):
    total_sentences = 0
    for simplified_version in simplified_versions:
        doc = nlp(simplified_version)
        total_sentences += len(list(doc.sents))

    return total_sentences


def get_sentence_length_stats(complex_sentences, simple_sentences):
    sentence_lengths = [get_sentence_length(sentence) for sentence in complex_sentences]
    average_num_sentences = []
    for simplified_versions in simple_sentences:
        if len(average_num_sentences) % 100 == 0:
            print(f"Processing sentence {len(average_num_sentences) + 1}...")
        average_num_sentences.append(get_num_sentences(simplified_versions) / len(simplified_versions) if simplified_versions else 0)

    return sentence_lengths, average_num_sentences
    
def sentence_stats_to_csv(complex_sentences, sentence_lengths, average_num_sentences):
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(f'{RESULTS_DIR}/sentence_stats_{date}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Length', 'Average Number of Sentences', 'Complex Sentence'])
        for i, sentence in enumerate(complex_sentences):
            writer.writerow([sentence_lengths[i], average_num_sentences[i], sentence])
    print(f"Results saved to {f.name}")
    return f'sentence_stats_{date}.csv'

def line_of_best_fit(csv_file):
    lengths = []
    avg_sentences = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths.append(float(row['Length']))
            avg_sentences.append(float(row['Average Number of Sentences']))

    slope, intercept, r_value, p_value, std_err = linregress(avg_sentences, lengths)

    print("Line of Best Fit: y = mx + b")
    print(f"m (slope): {slope}")
    print(f"b (intercept): {intercept}")
    print(f"R-squared: {r_value ** 2}")

    return slope, intercept, r_value ** 2

def generate_most_split_simp_file(complex_sentences, simple_sentences, set = 'valid'):

    filename = f'/Users/robertomilanesejr/MyDrive/USMC/NPS/Quarter7/Thesis/thesis-project/data/corpora/asset-main/dataset/asset.{set}.simp.most_split'
    complex_sentences, simple_sentences = get_sentence_pairs(ASSET_DIR, set='test')
    most_splits = []
    for simple_sentence in simple_sentences:
        max = 0
        max_sentence = None
        for sentence in simple_sentence:
            num_sentences = count_sentences(sentence)
            if num_sentences > max:
                max = num_sentences
                max_sentence = sentence
        most_splits.append(max_sentence)

    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in most_splits:
            f.write(sentence + '\n')

    print(f"Most split simplifications saved to {filename}")

def get_simple_most_split():
    filename = f'{ASSET_DIR}/asset.valid.simp.most_split'
    with open (filename, 'r', encoding='utf-8') as f:
        return f.readlines()


def get_depth_stats(sentences):
    max_depths = []
    i = 0
    for sentence in sentences:
        if i % 100 == 0:
            print(f"Processing sentence {i + 1}...")
        doc = nlp(sentence)
        max_depth = depth_tree_analysis(doc)
        max_depths.extend(max_depth)
        i += 1

    return max_depths

def get_depth_histogram():
    complex_sentences, simple_sentences = get_sentence_pairs(ASSET_DIR)
    simple_sentences = get_simple_most_split()

    complex_depths = get_depth_stats(complex_sentences)
    simple_depths = get_depth_stats(simple_sentences)

    print(f'Maximum depth of complex sentences: {max(complex_depths)}')
    print(f'Average depth of complex sentences: {np.mean(complex_depths)}')
    print(f'Standard deviation of complex sentence depths: {np.std(complex_depths)}')

    print(f'Maximum depth of simple sentences: {max(simple_depths)}')
    print(f'Average depth of simple sentences: {np.mean(simple_depths)}')
    print(f'Standard deviation of simple sentence depths: {np.std(simple_depths)}')


    # Create histograms
    plt.figure(figsize=(10, 6))
    plt.hist(complex_depths, bins=range(1, max(complex_depths) + 2), alpha=0.5, label='Complex Sentences', edgecolor='black')
    plt.hist(simple_depths, bins=range(1, max(simple_depths) + 2), alpha=0.5, label='Simple Sentences', edgecolor='black')

    # Labels and legend
    plt.xlabel('Syntactic Tree Depth')
    plt.ylabel('Frequency')
    plt.title('Histogram of Syntactic Tree Depth for Complex and Simple Sentences')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()





if __name__ == '__main__':
    complex_sentences, simple_sentences = get_sentence_pairs(ASSET_DIR)
    generate_most_split_simp_file(complex_sentences, simple_sentences, set='test')

        






