import pymupdf
import stanza

def read_pdf(file):
    '''Reads a pdf file and returns the text inside'''
    pdf = pymupdf.open(file)
    text = ''
    for page in pdf:
        text += page.get_text()
    return text

def split_sentences(text):
    '''Splits blck text from a file into a list of sentences'''
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    doc = nlp(text)
    sentences = []
    for sentence in doc.sentences:
        sentences.append(' '.join([word.text for word in sentence.words]))
    return sentences

def process_pdf(file):
    text = read_pdf(file)
    sentences = split_sentences(text)
    return sentences


def process_file(file):
    if file.endswith('.pdf'):
        return process_pdf(file)
    else:
        raise ValueError('File type not supported')