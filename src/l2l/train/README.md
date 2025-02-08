# Train the L2L model
## Generating test data

First get COCA.zip from our local DARPA-ARC Teams Files area.  Users not at NPS will have to request this from the COCA
developer.  Save COCA.zip to $CORPORA/COCA.zip and unzip. We need to generate word co-occurance data from COCA. Run

```
java  -Xmx40g -classpath   $ONTOLOGYPORTAL_GIT/sigmanlp/build/sigmanlp.jar:$ONTOLOGYPORTAL_GIT/sigmanlp/build/lib/* com.articulate.nlp.corpora.COCA -a
```

This will save files nouns.txt and verbs.txt in $CORPORA/COCA

Then generate language-logic pairs with GenSimpTestData. For 1M pairs, do

```
java -Xmx40g -classpath $ONTOLOGYPORTAL_GIT/sigmanlp/build/sigmanlp.jar:$ONTOLOGYPORTAL_GIT/sigmanlp/build/lib/* com.articulate.nlp.GenSimpTestData -s out 10000000
```

which will generate out-eng.txt and out-log.txt. Then generate the language logic pairs from all SUMO relations and all SUMO axioms with -

```
java -Xmx14g -classpath $ONTOLOGYPORTAL_GIT/sigmanlp/lib/*:$ONTOLOGYPORTAL_GIT/sigmanlp/build/classes com.articulate.nlp.GenSimpTestData -a allAxioms
java -Xmx60g -classpath $ONTOLOGYPORTAL_GIT/sigmanlp/lib/*:$ONTOLOGYPORTAL_GIT/sigmanlp/build/classes com.articulate.nlp.GenSimpTestData -g groundRelations
```

Concetenate these files together

```
cat allAxioms-eng.txt groundRelations-eng.txt out-eng.txt > combined-eng.txt
cat allAxioms-log.txt groundRelations-log.txt out-log.txt > combined-log.txt
```

### Execution time

On a 16 core fast laptop:

* 10,000 pairs takes 1.5 minutes
* 100,000 pairs takes 12.5 minutes
* 1,000,000 pairs takes 122 minutes


# T5 Model Training

## Instructions

## Step 0: Preprocess the data
On the command line, navigate to the foler where `combined-eng.txt` and `combined-log.txt` are located. Run the following command

`bash $ONTOLOGYPORTAL_GIT/sumonlp/scripts/preprocess/preproc1.sh`

## Step 1: Prepare Data

- Open the `$ONTOLOGYPORTAL_GIT/sumonlp/src/l2l/train/scripts/select_sentences.sh`
at `line 8` and `line 9` you can specify the path for the data (which in our case are the `combined-eng.txt-0` and `combined-log.txt-0`).
The `-0` means that the data have been preprocessed.

- run the command: `./select_sentences <number_of_shuffled_lines>`
if no `<number_of_shuffled_lines>` is specified it will use the whole dataset.
If `<number_of_shuffled_lines>` exist as arguments, it will create a training data file with the specified number of sentences.

- In both cases, the script is going to create the `training` and the `validating` data, where the validating data will be 10% of the testing data.

## Step 2: Train the model & Create the vocabulary DB.

### Train the model:

- Change the paths in the `batch_train.sh` file, and the resources if you want.

- Open the `train.py` and change the paths, so that they will point to your new create data from `Step 1`.

``` txt
# Paths to your data
input_file = 'data/full_12m_sentences/input_sentences.txt'
output_file = 'data/full_12m_sentences/output_logical.txt'
tokenized_output_file = 'data/full_12m_sentences/tokenized_data.json'
```
While in the same folder where your input_sentences.txt and output_logical.txt sentences are located, execute the batch_train.sh file:

```bash $ONTOLOGYPORTAL_GIT/sumonlp/src/l2l/train/batch_train.sh```


For the `tokenized_output_file`, you just need to specify the name of the file and the location that you want it to be saved. It should be empty, if you run the program for the first time.
If you have run the program again in the past for **the same dataset** you can comment out the
``` python
tokenize_data(input_file, output_file, tokenized_output_file)
```
to reduce time.

### Create the vocabulary DB.

IF YOU TRAINED A MODEL ON THE **SAME** TRAINING DATA, THEN YOU CAN USE THE `vocabulary.db` CREATED FOR THE PREVIOUS ONE.

Now that the `training data` has been generated, it can be used to create the Vocabulary.db for this model.

- Open the `/data/scripts/create_voc_from_sentences.py` file.
- Change the `DB_PATH` at line 5, to the location you want to save the DB.
- Change the `SENTENCE_PATH` at line 6, to the location that the `training data` is.
- run the batch file `create_voc_from_sentences.sh` located in the scripts folder.



## Step 3: Validate models

### Validate one model.

- Change again the `log paths` in the `evaluate_model_job.sh`.
- Open the `evaluate_model.py` and:
1. Change the path of the model you want to test.
``` python
    model = load_model('data/500k_sentences_suffled/t5_model')
```
2. Change the path for the validation data at lines **53** and **58**.

``` python
    with open('data/500k_sentences_suffled/input_sentences_500k_val.txt', 'r') as f:
    ....
    with open('data/500k_sentences_suffled/output_logical_500k_val.txt', 'r') as f:
```

- Run the `./evaluate_model_job.sh`
It will create a new file called `predictions_and_references.txt`, and it will save each prediction and its true value for every sentence that the model will be validated on, and in the end it will print the percentage of the prediction that are **exactly** the same with the true values.

### Testing and Comparing multiple models on Custom Data.

- Open the file `multiple_models_reference.py`.
Change the paths in the array at line **24**, with a value for every model you want to test.

``` python
    model_paths_names = [
      ('data/full_12m_sentences/t5_model_3_epochs','12m Sentences'),
      ('data/500k_sentences_suffled/t5_model','500k SUFFLED Sentences'),
      ('data/500k_last_sentences/t5_model','500k Last Sentences')
    ]  # Add your actual model paths here
```

- write the `sentence`  you want to test in the `input.txt` file.

- Run the python file
`python3 multiple_models_reference.py`
A new `output_multiple_models.txt` will be created with the predictions for each model, for easy comparison.
Example:

``` text
Input: OrangeRed is an instance of Orange

Model: 12m Sentences
Output: ( instance OrangeRed Orange )

Model: 500k SUFFLED Sentences
Output: ( instance OrangeRed Orange )

Model: 500k Last Sentences
Output: ( exists (? H? P? DO? IO ) ( and ( instance? H Human ) ( names "OrangeRed"? H ) ( instance? P Ingesting ) ( experiencer? P? H ) ( attribute? DO Female ) ( names "OrangeRed"? DO ) ( instance? DO Human ) ( objectTransferred? P? DO ) ) )

==================================================
```







