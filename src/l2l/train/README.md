## Setup Environment

This project is utilizing pixi dependency manager. Every package and its version is specified in the **pixi.toml** file.

### STEP 1: Install dependencies

1. After connecting to Hamming, install Pixi if not already installed:

   `curl -fsSL https://pixi.sh/install.sh | bash`
2. Start a session on Hamming with a GPU. For example:

   `salloc -p genai --gres=gpu:1`
3. While in the new_train folder, install the environment with the following command:

   `pixi install`

   It is important to be on a node with GPU(s) allocated. Otherwise, the above will not work.
4. Exit the GPU session:

   `exit`
5. For sbatch scripts, add the following before the execution of your python script so that
   the pixi environment is activated:

   `eval "$(pixi shell-hook -s bash)"`

### STEP 2: Prepare data

1. Data must be in JSON format. Open the `scripts/txt_to_json.sh` file and change the paths for the english and logic sentences.
2. Run the txt_to_json.sh script. The new json file will be saved in the path of the script.
3. Move the dataset, to the folder you prefer. I recomend moving it into the /data folder for consistency.

### STEP 3: Change data paths for Training the model & Create the vocabulary DB.

#### Train the model.

1. Open train.sh script which is located inside the /scripts/ folder and change the `paths.input_file` value, to the new json dataset that you've generated in STEP 2.
2. At the same scirpt change the `paths.data_name` value, with the filename of the new dataset. This value will be used to check if the dataset has been already tokenized in the past, in order to skip doing it again.

#### Create the vocabulary.db
IF YOU TRAINED A MODEL ON THE **SAME** TRAINING DATA, THEN YOU CAN USE THE `vocabulary.db` CREATED FOR THE PREVIOUS ONE.

We will use the training data to create the Vocabulary.db.

- Open the `/scripts/create_voc_from_sentences.py` file.
- Change the `DB_PATH` at line 5, to the location you want to save the DB. Be carefull not to overide other usefull databases that you may need.
- Change the `SENTENCE_PATH` at line 6, to the location that the natural language `training data` is. We need the text file that contains the natural language and not the JSON one.
- run the batch file `create_voc_from_sentences.sh` located in the scripts folder.

### STEP 4: Tune the Hyperparams

1. You can check the default Hyperparams in the `configs/configs.yaml` file.
2. You can **override** some of these hyperparams in the `train.sh` script.

### STEP 5: Train the model

1. Run the train.sh script.

### STEP 6: Model Evaluation.

Check the model:

- There are 2 ways to test the model.

1. The first one is by open a chat with it where you can write the sentence, and the model will return the logic.
   To do that open the `chat_inference.py` file and change the value of **CHECKPOINT_PATH** variable to point to the new checkpoint you want to check. Then just run the script.
2. The second one is by using the `inference.py` where there are predefined sentences in the script and by running it it will create an `output.csv` file with all the responces.
   Again to use that, you have to change the value of `model_path` in the file to point to the new checkpoint. Then just run the script.

## Project packages:

- pytorch
- lightning
- torchmetrics
- torchvision
- matplotlib
- pandas
- numpy
- pyrootutils
- hydra-core
- pytorch-cuda
- webdataset
- scikit-learn

More detais about the versions of each package you can find in the **pixi.toml**
