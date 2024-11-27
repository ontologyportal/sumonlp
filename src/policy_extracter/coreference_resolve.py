import stanza
import stanza.models.coref as coref
import logging
import datetime
import platform
import os



class CoreferenceResolver:
    def __init__(self):
        self.coref_logger = self.initialize_logger()
        self.pipe = stanza.Pipeline('en', processors='tokenize,coref,pos,ner,mwt,lemma,depparse', use_gpu=True)


    def initialize_logger(self):
        # Create a dedicated logger for coreference_resolver
        logger = logging.getLogger('coreference_resolver')
        logger.setLevel(logging.INFO)  # Set log level

        # Check if handlers are already added to avoid duplicate logs
        if not logger.handlers:
            # Create a file handler
            log_file = 'logs/coreference.log'
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

    def resolve_coreference(self, data):
        '''Resolves coreferences in a given sentence or paragraph'''
        doc = self.pipe(data)

        resolved_sentence = ''

        if False:
            print(*[f'token: {token.text}\tner: {token.multi_ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')
            print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
            print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head is not None and word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
    

        for sentence in doc.sentences:
            for word in sentence.words:
                if False:
                    token_dict = word.to_dict()
                    if "coref_chains" in token_dict:
                        print(f"Token: {token_dict['text']}")
                        
                        for coref_chain in token_dict["coref_chains"]:
                            print(dir(coref_chain))  # List all methods and attributes
                            print(vars(coref_chain))  # Show the internal dictionary (if available)

                if word.coref_chains != [] and word.upos == 'PRON':
                    self.coref_logger.info(f'Coreference found: {word.text}. Resolving with {word.coref_chains[0].chain.representative_text}')
                    coref_chains = word.coref_chains
                    coref_chain = coref_chains[0]
                    if 'Poss=Yes' in word.feats:
                        self.coref_logger.info(f'Possessive pronoun found: {word.text}.')
                        resolved_sentence += coref_chain.chain.representative_text + "'s "
                    else:
                        resolved_sentence += coref_chain.chain.representative_text + ' '
                else:
                    if word.text in ['.', ',', '!', '?', "'s", ':', ';']:
                        resolved_sentence = resolved_sentence[:-1] + word.text # Replace the last character with the token
                        resolved_sentence += ' '
                    elif word.text in ['-', '–']:   #hyphenated word, no space
                        resolved_sentence = resolved_sentence[:-1] + word.text # Replace the last character with the token
                    else:
                        resolved_sentence += word.text + ' '

        data = data.strip()
        resolved_sentence = resolved_sentence.strip()
        
        if data != resolved_sentence:
            self.coref_logger.info(f'Original Sentence: {data}')
            self.coref_logger.info(f'Resolved Sentence: {resolved_sentence}')
        else:
            self.coref_logger.info(f'No coreferences resolved in the sentence: {data}')
            
        return resolved_sentence

    


