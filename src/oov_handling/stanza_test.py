import stanza
from stanza.pipeline.core import DownloadMethod

stanza.download('en', processors='tokenize,pos', logging_level='WARN')

nlp = stanza.Pipeline('en', processors='tokenize,pos', download_method=DownloadMethod.REUSE_RESOURCES, logging_level='WARN')

doc = nlp('Barack Obama was born in Hawaii.')

for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, word.lemma, word.pos)

