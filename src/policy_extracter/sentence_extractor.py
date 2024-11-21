import pymupdf
import stanza
import logging
import datetime
import platform
import os

class SentenceExtractor:
    '''Extracts sentences from a file'''

    def __init__(self):
        stanza.download('en')
        self.pipe = stanza.Pipeline(lang='en', processors='tokenize')
        self.extract_logger = self.initialize_logger()

    def initialize_logger(self):
        # Create a dedicated logger for sentence_extractor
        logger = logging.getLogger('sentence_extractor')
        logger.setLevel(logging.INFO)  # Set log level

        # Check if handlers are already added to avoid duplicate logs
        if not logger.handlers:
            # Create a file handler
            log_file = 'logs/extractor.log'
            os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure directory exists
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Create a formatter and set it for the handler
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(file_handler)

        # Log initialization details
        start_time = datetime.datetime.now()
        logger.info("Program starting")
        logger.info(f"Start time: {start_time}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Current working directory: {os.getcwd()}")

        return logger
    
    def read_pdf(self, file):
        '''Reads a pdf file and returns the text inside'''
        pdf = pymupdf.open(file)
        text = ''
        for page in pdf:
            text += page.get_text()
        text = text.replace('-\n', '')
        text = text.replace('\n', ' ')
        return text

    def split_sentences(self, text):
        '''Splits blck text from a file into a list of sentences'''

        doc = self.pipe(text)
        sentences = []
        for sentence in doc.sentences:
            sentences.append(sentence.text)
        return sentences

    def process_pdf(self, file):
        '''Processes a PDF file and returns a list of sentences'''
        text = self.read_pdf(file)
        self.extract_logger.info(f'Extracted text from PDF: {text}')

        sentences = self.split_sentences(text)
        self.extract_logger.info(f"Number of sentences: {len(sentences)}")
        for i, sentence in enumerate(sentences):
            self.extract_logger.info(f"Sentence {i}: {sentence}")
        return sentences


    def process_file(self, file):
        if file.endswith('.pdf'):
            self.extract_logger.info(f"Processing PDF file: {file}")
            return self.process_pdf(file)
        else:
            raise ValueError('File type not supported')