import coreference_resolve
import sentence_extractor

def get_files():
    '''Get all files from the input file'''
    import os
    with open('input_pe.txt', 'r') as f:
        files = f.readlines()
    files = [f.strip() for f in files]
    return files


def process_file(file_path, se, cr):
    '''Process a single file and resolve coreferences.'''
    # Extract sentences from the file
    sentences = se.process_file(file_path)
    print(f"Number of sentences: {len(sentences)}")

    # Resolve coreferences
    resolved_sentences = resolve_coreferences(sentences, se, cr)
    return resolved_sentences

def resolve_coreferences(sentences, se, cr):
    '''Resolve coreferences for a list of sentences.'''
    resolved_sentences = []
    
    # Process the first two sentences first. 
    current = sentences[:2]
    sentences = sentences[2:]
    resolved_sentences.extend(process_chunk(current, se, cr))

    # Process remaining sentences iteratively
    while sentences:  # While there are sentences left
        for i, sentence in enumerate(resolved_sentences):
            cr.coref_logger.info(f"Resolved sentence {i}: {sentence}")
        last_resolved = resolved_sentences[-1]      # Get the last resolved sentence
        first_sentence = sentences.pop(0)         # Get the first sentence from the remaining sentences
        current = [last_resolved, first_sentence]   # Combine the last resolved sentence and the first unresolved sentence
        resolved_chunk = process_chunk(current, se, cr)  # Process the chunk
        if len(resolved_chunk) == 2:
            resolved_sentences.append(resolved_chunk[1])   # resolve them and skip the first sentence as it is already resolved
        else:     # there should only be two sentences in the chunk, if not, the extracted text is likely undecipherable to stanza. This can happen with footnotes, citations, etc.
            pass
    
    return resolved_sentences

def process_chunk(chunk, se, cr):
    '''Process a chunk of sentences and resolve coreferences''' 
    current = ' '.join(chunk)
    print(f"Processing chunk: {current}")
    current = cr.resolve_coreference(current)
    return se.split_sentences(current)


if __name__ == '__main__':
    # Initialize dependencies
    se = sentence_extractor.SentenceExtractor()
    cr = coreference_resolve.CoreferenceResolver()

    # Get all files
    files = get_files()
    for file_path in files:
        process_file(file_path, se, cr)

