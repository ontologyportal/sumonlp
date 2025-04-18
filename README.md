# Overview
SUMO Natural Language Processing (SUMO NLP) takes natural language text and turns the text into logical statements. These statements can be queried against, and tested for inconsistencies.

# Install pre-requisites
SUMO NLP is reliant on the the following:

* SUMO and SIGMAKEE. Installation instructions are [here](https://github.com/ontologyportal/sigmakee).
* Vampire. Installation instructions are [here](https://github.com/ontologyportal/sigmakee?tab=readme-ov-file#vampire). Configuration instructions below.
* Miniconda (instructions below)
* Ollama (instructions below)

## Create folder structure
Create a sumonlp folder where models and large files will be stored. This should be separate from where you clone this repository. For example:

```
cd ~
mkdir .sumonlp
cd .sumonlp
mkdir L2L_model
```

Add the following lines to your .bashrc file. HAMMING is a super computer at the Naval Postgraduate School. If you are not running on the Hamming super computer, this should be set to false:

```
export SUMO_NLP_RUNNING_ON_HAMMING=false
export SUMO_NLP_HOME=~/.sumonlp
```

Clone the repository to your workspace directory:
```
cd ~/workspace
git clone https://github.com/ontologyportal/sumonlp.git
```

## Miniconda
Follow install instructions at https://docs.anaconda.com/miniconda/install/ . Linux instructions reproduced here for convenience:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

Create a conda environment. This allows you to create, export, list, remove, and update environments that have different versions of Python and/or packages installed in them.

As an example: If your version of Python were 3.9, then you would run:
```
conda create -n name_you_choose_for_environment python=3.9
```

Add the name of your environment ("name_you_choose_for_environment") to the .bashrc file:

```
export SUMO_NLP_CONDA_ENVIRONMENT="name_you_chose_for_environment"
```

## Vampire Prover
Installation instructions are [here](https://github.com/ontologyportal/sigmakee?tab=readme-ov-file#vampire). It is recommended to install vampire in After installing Vampire, add the following line to the .bashrc file:

```
export PATH="/path/to/vampire:$PATH" 
```

### Configuration
The language to logic translator produces the file SUMO_NLP.kif, with logic statements generated from natural language. This file is automatically copied to the .sigmakee/KBs directory. To be able to translate the generated SUO-KIF logic to the TPTP input required by the vampire prover, update 

```
$SIGMA_HOME/KBs/config.xml
```

and add the following line is in the <kb name="SUMO" > section:

```
...
  <kb name="SUMO" >
    ...
    <constituent filename="$SIGMA_HOME/KBs/SUMO_NLP.kif" />
    ...
  </kb>
...

```

The files within this tag are all combined and translated into the SUMO.fof file which is then fed as input to the vampire prover.

## Ollama Install Notes

Instructions adapted from: 
https://github.com/ollama/ollama/blob/main/docs/linux.md
https://ollama.com/download/linux


```
cd Programs
mkdir ollama
cd ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C ./ -xzf ollama-linux-amd64.tgz
cd bin
sudo chmod 777 *
```

To start a server, from the ollama/bin directory
```
./ollama serve
```


On different terminal, from the ollama/bin directory

```
./ollama -v
```

If you see a version, then it worked!

To run a specific model on the server (and download if necessary). For example:

```
./ollama run llama3.2 
```

For a list of models - https://ollama.com/library

Add the path of your ollama installation to the .bashrc file:

```
export OLLAMA_HOST_PORT="11434" # Used to change default port (11434) to unique port number if necessary.
export OLLAMA_HOST="127.0.0.1:$OLLAMA_HOST_PORT"
export PATH="/path/to/ollama/bin:$PATH" # Path to ollama executable
```


## L2L model and vocabulary.db
The vocabulary.db file must be present, as well as the model used for language to logic conversion.

For more information, see [here](https://github.com/ontologyportal/sumonlp/blob/master/src/l2l/train/README.md)

Place the vocabulary.db file in $SUMO_NLP_HOME location, and model used for L2L conversion to $SUMO_NLP_HOME/L2L_Model.

# Running sumonlp
1. To configure the models that will be run during different parts of the pipeline, edit the src/load_configs.sh file.
2. Run src/utils/install_requirements.sh. This will install necessary python packages.
3. Run src/main.sh
4. Type "help" to list available commands.

## Running on Hamming GPU Node:

From a submit node, run

srun --pty -N 1 --partition=genai --gres=gpu:1 bash
