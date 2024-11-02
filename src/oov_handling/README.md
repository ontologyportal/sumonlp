## Out of Vocabulary Detection:

### How to run:
The python program is expecting to read a document saved in  `/home/angelos.toutsios.gr/workspace/sumonlp/src/oov_handling/input_oov.txt` file.

The output will be in the `/home/angelos.toutsios.gr/workspace/sumonlp/src/oov_handling/output_oov.txt` file.

```sh
python3 -u oov_handling.py
```

### How it works!

- The program identifies all noun, phrase-nouns (names, etc) and verbs and searches for their **root** word in the vocabulary.
- The vocabulary consists of all the **root** words from the COCA documents.
- If the word does not exist in the vocabulary then this word is saved in a table in the DB and an ID is assigned to it. This word in the text is replaced by the string **<UNK_WordType_Id>**
- The sentences after all the replacements are been saved in the output file for further processing.



