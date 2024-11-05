#!/bin/bash
#SBATCH --job-name=sumo_nlp_training_smallwork_1000
#SBATCH --output=/home/roberto.milanese/scripts/log_%j.out  # Output log file
#SBATCH --error=/home/roberto.milanese/scripts/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00             # Time limit (hh:mm:ss)
#SBATCH --partition=genai
#SBATCH --gres=gpu:1               # Request 1 GPU
 
# Activate environment
# module load lib/cuda/9.0.176
# module load util/cuda-toolkit/12.0
module load lang/java/8.141-oracle
javac LlamaMTrans.java
# conda activate nmt_env  # Change to your desired conda environment
 
# Start Ollama server in the background and log its output
export OLLAMA_HOST="127.0.0.1:55757"
ollama serve > /home/jarrad.singley/Programs/llamaMD/ollama_log.out 2>&1 &
 
# Capture the PID of the Ollama server so we can track/kill it later if needed
OLLAMA_PID=$!
 
# Sleep to ensure Ollama has started
sleep 5
 
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull mistral
echo "running detector"
python3 metaphor_detect_pipeline.py
echo "running translator"
# Run the Python script
java LlamaMTrans output_md.txt output_mh.txt

# send output to input of sentence simplifier
cp output_mh.txt ../sentence_simplification/input_ss.txt
# Optionally, kill the Ollama server when the script finishes
kill $OLLAMA_PID