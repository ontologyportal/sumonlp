#Train the L2L model

First get COCA.zip from our local
DARAPA-ARC Teams Files area.  Users not at NPS will have to request this from the COCA
developer.  We need to generate word co-occurance data from COCA. Run

java  -Xmx40g -classpath   $ONTOLOGYPORTAL_GIT/sigmanlp/build/sigmanlp.jar:$ONTOLOGYPORTAL_GIT/sigmanlp/build/lib/* com.articulate.nlp.corpora.COCA -a

This will save files nouns.txt and verbs.txt in $CORPORA/COCA

Then Generate language-logic pairs with GenSympTestData. For 1M pairs, do

```
java  -Xmx40g -classpath   $ONTOLOGYPORTAL_GIT/sigmanlp/build/sigmanlp.jar:$ONTOLOGYPORTAL_GIT/sigmanlp/build/lib/* com.articulate.nlp.GenSimpTestData -s out 10000000
```

which will generate out-eng.txt and out-log.txt

## Execution time

On a 16 core fast laptop: 

* 10,000 pairs takes 1.5 minutes
* 100,000 pairs takes 12.5 minutes
* 1,000,000 pairs takes 122 minutes

Then tokenize the data by calling sumonlp/src/l2l/train/tokenize_data.py with

```
tokenize_data('out-eng.txt', 'out-log.txt', 'tokenized_data.json')
```

This line appears in train.py commented out

Put these files in a 'data' subdirectory below where you call train.py along with renaming
out-eng.txt to data/input_sentences.txt and out-log.txt to data/output_logical.txt

A CUDA OutOfMemory error meant I changed train.py line 35 to batch_size=16 instead of 32

10 hours to train on 1M pairs in 3 epochs to an average loss of 0.0085

242MB model file is the result.

The model gets saved under the same directory where one puts the input/output pairs and tokens,
as full_12m_sentences/t5_model_3_epochs .  We should consider changing the directory names
and file names, for example making the name "full_12m_sentences" a parameter in model.py instead of a hard coded name.




