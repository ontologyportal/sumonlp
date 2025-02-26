from sentence_transformers import SentenceTransformer
import spacy
from spacy import displacy
import faiss
import numpy as np
from zss import simple_distance, Node
import os
import asset_analyzer



ASSET_DIR = './data/corpora/asset-main/dataset'
EMBEDDINGS_DIR = './data/embeddings'
CUSTOM_DIR = './data/custom'


MODEL = 'all-MiniLM-L6-v2'
EMBEDDINGS_FILE = f'{EMBEDDINGS_DIR}/asset_embeddings.npy'
FAISS_INDEX_FILE = f'{EMBEDDINGS_DIR}/faiss_index'
CLOSEST_TREES_FILE = f'{EMBEDDINGS_DIR}/tree_indices.csv'
STRUCTURAL_EMBEDDINGS_FILE = f'{EMBEDDINGS_DIR}/structural_asset_embeddings.npy'
STRUCT_FAISS_INDEX_FILE = f'{EMBEDDINGS_DIR}/faiss_index'


INDICES = [35, 109, 158, 251, 424, 608, 877, 916, 1692, 1776]
# shift all values in INDICES right by 1 to match 1-indexed sentences in the dataset
INDICES = [i-1 for i in INDICES]


SIMP_FILE = f'{ASSET_DIR}/asset.valid.simp.most_split'
nlp = spacy.load('en_core_web_sm')

# ---------------------------------------------------------
# ----------------- Tree Similarity -----------------
# ---------------------------------------------------------
def build_tree(sentence, print_tree=False):
    '''Takes in a sentence and returns the root node of the zss dependency tree'''

    doc = nlp(sentence)

    # Find the root token
    root = [token for token in doc if token.head == token][0]

    tree = build_zss_tree(root)

    if print_tree:
        print(tree)

    return tree
    
def build_zss_tree(token):
    """Recursively build a ZSS tree from a spaCy dependency tree."""
    node = Node((token.dep_, token.pos_))
    for child in token.children:
        node.addkid(build_zss_tree(child))

    

    return node

def custom_node_cost(node1, node2):
    '''custom cost for comparing nodes of zss style (dep, pos)'''

    if not node1 or not node2:
        return 1  # full cost for missing nodes
    
    # Split the node label into its components
    dep1, pos1 = node1
    dep2, pos2 = node2

    # .25 cost if pos are same and dep are different, or vice versa for .75
    if dep1 != dep2 and pos1 == pos2:
        return .75
    if dep1 == dep2 and pos1 != pos2:
        return .25

    # full cost if both are different
    if dep1 != dep2 and pos1 != pos2:
        return 1
    else:
        return 0


def compute_tree_similarity(tree1, tree2):
    '''computes and returns the tree edit distance between two trees'''
    return simple_distance(tree1, tree2, label_dist = custom_node_cost)

def load_trees():
    '''load in already existing tree embeddings'''
    if os.path.exists(STRUCTURAL_EMBEDDINGS_FILE):
        trees = np.load(STRUCTURAL_EMBEDDINGS_FILE, allow_pickle=True)
        return trees
    print("No stored trees found.")
    return None

def save_trees(trees):
    '''save the parsed trees to file'''
    np.save(STRUCTURAL_EMBEDDINGS_FILE, trees)
    print(f"Successfully saved {len(trees)} trees.")

def add_new_sentences_to_index(sentences):
    '''Parse new sentences into trees and add unique ones to the existing tree embeddings.'''
    existing_trees = load_trees()
    new_trees = np.array([build_tree(s) for s in sentences], dtype=object)

    if existing_trees is not None:
        existing_tree_strings = {str(tree) for tree in existing_trees}
        
        # Filter out new trees that already exist
        unique_new_trees = [tree for tree in new_trees if str(tree) not in existing_tree_strings]

        if not unique_new_trees:
            print("No new trees to add.")
            return  # No need to save if nothing is new

        updated_trees = np.concatenate((existing_trees, unique_new_trees))
    else:
        updated_trees = new_trees

    save_trees(updated_trees)

def search_tree_similarity(query, top_k=5):
    '''Find the most structurally similar sentences using tree edit distance.'''
    stored_trees = load_trees()

    if stored_trees is None:
        raise ValueError("No stored trees found. Run `embed_structural_assets()` first.")

    query_tree = build_tree(query)

    distances = np.array([compute_tree_similarity(query_tree, t) for t in stored_trees])
    closest_indices = np.argsort(distances)[:top_k]

    return closest_indices, distances[closest_indices]

def embed_structural_assets():
    '''Parse all asset sentences into trees and store them.'''
    stored_trees = load_trees()
    if stored_trees is not None:
        print(f"Loaded {len(stored_trees)} stored trees.")
        response = input('Would you like to add new sentences to the index? (y/n): ')
        if response.lower()[0] == 'y':
            with open(f'{ASSET_DIR}/asset.valid.orig', 'r') as f:
                sentences = [line.strip() for line in f.readlines()]
            add_new_sentences_to_index(sentences)
    else:
        filepath = f'{ASSET_DIR}/asset.valid.orig'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f.readlines()]

        trees = np.array([build_tree(s) for s in sentences], dtype=object)
        save_trees(trees)
        print(f"Successfully embedded {len(trees)} sentences.")





# ---------------------------------------------------------
# ----------------- Syntactic Embeddings ------------------
# ---------------------------------------------------------


def embed_assets():
    '''embeds the asset sentences and saves the embeddings and the FAISS index to file'''
    model = SentenceTransformer(MODEL)
    with open(f'{ASSET_DIR}/asset.valid.orig', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]
    embeddings = model.encode(sentences, convert_to_numpy=True)  # generate sentence embeddings

    np.save(EMBEDDINGS_FILE, embeddings)  # save embeddings to file


    dimension = embeddings.shape[1]       # save the FAISS index
    index = faiss.IndexFlatL2(dimension)  # l2 distance index
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

# Function to load embeddings and perform search
def search_faiss(query, top_k=5):
    '''searches the nearest neighbors of the query sentence in the embeddings, returns the indices and distances of the top_k nearest neighbors'''
    model = SentenceTransformer(MODEL)
    query_embedding = model.encode([query], convert_to_numpy=True)

    index = faiss.read_index(FAISS_INDEX_FILE)
    distances, indices = index.search(query_embedding, top_k)   # search for top_k nearest neighbors

    return indices, distances

def get_sentence_pairs(query, top_k=5, context_type='dynamic_tree', return_indices=False):
    '''Searches the nearest neighbors of the query sentence in the embeddings, returns a k long list of tuples
    containing the original sentence and the corresponding simplified sentence of the nearest neighbors.'''
    if top_k <= 0:
        return None
    if context_type == 'dynamic_similarity':
        indices, distances = search_faiss(query, top_k)
        indices = indices.flatten().tolist()  # Ensure it's a flat list
    elif context_type == 'dynamic_tree':
        indices, distances = search_tree_similarity(query, top_k)
    elif context_type == 'static':
        indices = INDICES[:top_k]
    elif context_type == 'random':
        indices = np.random.choice(range(2000), top_k, replace=False).tolist()
    else:
        raise ValueError('Invalid context type. Must be one of "dynamic_similarity", "dynamic_tree", "static", or "random".')

    with open(f'{ASSET_DIR}/asset.valid.orig', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(SIMP_FILE, 'r') as f:
        simple_sentences = [line.strip() for line in f.readlines()]

    if return_indices:
        return indices
    
    return [(sentences[i], simple_sentences[i]) for i in indices]

def get_custom_sentence_pairs(top_k=5):
    '''returns a k long list of tuples containing the original sentence and the corresponding simplified sentence of the nearest neighbors.'''

    with open(f'{CUSTOM_DIR}/asset_custom.orig', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(f'{CUSTOM_DIR}/asset_custom.simp', 'r') as f:
        simple_sentences = [line.strip() for line in f.readlines()]

    if top_k > len(sentences):
        top_k = len(sentences)

    return [(sentences[i], simple_sentences[i]) for i in range(len(sentences))][:top_k]


def run_faiss_embeddings():
    '''Run the entire process of embedding the asset sentences allows searching for nearest neighbors.'''
    embed_assets()
    while(True):
        top_k = int(input('Enter the number of nearest neighbors to return: '))
        context_type = input('Enter the context type (dynamic, static, random): ')
        query = input('Enter the query sentence: ')
        sentence_pairs = get_sentence_pairs(query, top_k, context_type)

        for i, pair in enumerate(sentence_pairs):
            print(f'{i+1}. Original: {pair[0]}')
            print(f'   Simplified: {pair[1]}')
            print()


def tree_process_to_csv(sentences):
    '''parses the given sentences, finds the top 10 nearest tree neighbors for each and saves 
    the original and the indices of the top 10 tree edit distances to a csv file'''

    tree_indices = {}

    for i, sentence in enumerate(sentences):
        indices = get_sentence_pairs(sentence, 10, 'dynamic_tree', return_indices=True)
        tree_indices[i] = indices
        print(tree_indices)

    with open('tree_indices.csv', 'w') as f:
        for i, indices in tree_indices.items():
            f.write(f'{i}, {indices}\n')

def tree_process_batches(sentences, batch_size=10, csv_file='tree_indices.csv'):
    """
    Processes sentences in batches and writes their tree indices to a CSV file.
    
    If the CSV file already exists, this function resumes processing from the
    sentence immediately after the last processed one. It processes the sentences
    in batches of `batch_size` and flushes the output after each batch.

    Parameters:
    - sentences: List of sentences to process.
    - batch_size: Number of sentences to process before flushing to file.
    - csv_file: Name/path of the CSV file for storing results.
    """

    # Determine starting index by checking for an existing CSV file.
    start_index = 0
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # Assume each line is formatted as "index, <indices>"
                try:
                    last_line = lines[-1].strip()
                    last_index = int(last_line.split(',')[0])
                    start_index = last_index + 1
                    print(f"Resuming from sentence index: {start_index}")
                except (ValueError, IndexError):
                    print("Could not parse the last line of the CSV. Starting from index 0.")
                    start_index = 0

    # Open the CSV file in append mode if resuming, otherwise write mode.
    file_mode = 'a' if start_index > 0 else 'w'
    with open(csv_file, file_mode) as f:
        batch_count = 0
        # Process sentences starting from the determined start_index.
        for i in range(start_index, len(sentences)):
            sentence = sentences[i]
            # Obtain the top 10 tree neighbors (or indices) using your helper function.
            indices = get_sentence_pairs(sentence, 10, 'dynamic_tree', return_indices=True)
            
            # Write the result for this sentence to the CSV.
            f.write(f'{i}, {indices}\n')
            batch_count += 1

            # Flush the file every time a batch is complete.
            if batch_count >= batch_size:
                f.flush()
                print(f'Batch written for sentences {i - batch_size + 1} to {i}.')
                batch_count = 0

        # In case there is an incomplete batch at the end, flush it.
        if batch_count > 0:
            f.flush()
            print(f'Final batch written up to sentence index {len(sentences) - 1}.')

    print('Processing complete.')



if __name__ == '__main__':
    while True:
        response = input('Enter query sentence: ')
        if response == 'exit':
            break
        sentence_pairs = get_sentence_pairs(response, 5, 'dynamic_tree')

        for i, pair in enumerate(sentence_pairs):
            print(f'{i+1}. Original: {pair[0]}')
            print(f'   Simplified: {pair[1]}')
            print()
