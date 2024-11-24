import stanza
# from stanza.pipeline.core import DownloadMethod

# Custom packages
ner_packages = ["ncbi_disease", "ontonotes-ww-multi_charlm","linnaeus","bionlp13cg"]
nlp1 = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,ner', package={"ner":ner_packages })

# Default packages
nlp2 = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,ner')


# paragraph = "In the humid rainforest climate of the U.S.A Basin, the mighty Amazon River flows through multiple South American countries, including Brazil and Peru, supporting diverse ecosystems. Not far, the Andes Mountains stretch along the continent, home to species like the Andean condor and the spectacled bear. Meanwhile, in North America, the National Park Service protects regions like Yellowstone National Park, where geothermal features like Old Faithful draw visitors. Over in Europe, the Rhône River cuts through France, winding near the Swiss Alps’ Matterhorn peak. Elsewhere, environmental NGOs like the World Wildlife Fund and Greenpeace focus on preserving biodiversity hotspots, including the Congo Basin and the Mekong River Delta, which face threats from climate change-induced flooding and droughts. Back in Asia, Mount Fuji towers over Japan, while the Gobi Desert spans China and Mongolia, experiencing extreme temperature swings. The Red Cross and the United Nations Climate Change Secretariat continue efforts to address global warming and its effects, with international conferences often held in cities like Geneva, Nairobi, and New York and has hip arthritis."
paragraph = '''Python and Java are computer languages'''
doc1 = nlp1(paragraph)
doc2 = nlp2(paragraph)
# doc = nlp("The Baobab trees in Madagascar and the Giant Sequoias in California are some of the oldest living species on Earth.")
# doc = nlp("Residents of San Pedro de Atacama near Mount Vesuvius were warned about the coming snowstorm on Elm Street.")
# doc = nlp("The Eucalyptus trees in Australia are known for their distinctive aroma and tall height.")
# doc = nlp("Scientists discovered a rare species of Welwitschia in the deserts of Namibia, where it survives with minimal rainfall.")
# doc = nlp("John Bauer works at Stanford and has hip arthritis.  He works for Chris Manning")



# print('------------------------------------------------------------')
# print(*[f'word: {word.text}\t word_lemma: {word.lemma}\t word_pos: {word.upos}' for sent in doc.sentences for word in sent.words], sep='\n')

print('------------------------------------------------------------')
print('---------  Custom Packages  ---------------')
# print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc1.ents], sep='\n')
# print(*[f'token: {token.text}\tner: {token.multi_ner}' for sent in doc1.sentences for token in sent.tokens], sep='\n')
print('------------------------------------------------------------')
print('---------  Default Packages  ---------------')
print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc2.ents], sep='\n')
print('------------------------------------------------------------')
print('---------  Word Type  ---------------')
categories = {'NOUN', 'VERB', 'PROPN'}  # replace with your desired categories
# print(*[f'word: {word.text}\t Type: {word.upos}'
#         for sent in doc1.sentences
#         for word in sent.words
#         if word.upos in categories], sep='\n')

# print(*[f'token: {token.text}\tner: {token.multi_ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')
# print(*[f'token: {token}' for sent in doc.sentences for token in sent.tokens], sep='\n')

# print('------------------------------------------------------------')
# print(doc.ents)
# print('------------------------------------------------------------')
