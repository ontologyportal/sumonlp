#!/bin/bash
##############################################################
# When running this, run with
#
#   source config_path.sh
#
# This ensures it applies to the current batch session.
# Otherwise when the script is run, the PATH is only valid
# for the commands in config_path.sh
###############################################################


################### UPDATE THESE ####################
export MODEL_L2L="t5_model"
export MODEL_MH="mistral" # Model to be used in metaphor handling. Common models are mistral and llama3.2
export MODEL_PP="llama3.2" # Model to be used in PPara. Common models are mistral and llama3.2
export MODEL_SS="llama3.1:8b-instruct-q8_0" # Model to be used in sentence simplification. Common models are mistral and llama3.2
export MODEL_CHECKPOINT="/data/fsg/Trained_Models/last.ckpt" # Model to be used in checkpointing. Common models are mistral and llama3.2
######################################################




################## DON'T TOUCH THESE ######################
#initialize the conda environment
eval "$(conda shell.bash hook)"
conda activate "$SUMO_NLP_CONDA_ENVIRONMENT"
