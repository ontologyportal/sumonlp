#!/bin/bash
#SBATCH --job-name=sumo_weirdness_detector_on_training_data
#SBATCH --output=./logs/log_%j.out  # Output log file
#SBATCH --error=./logs/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=64
#SBATCH --time=80:00:00              # Time limit (hh:mm:ss)
#SBATCH --partition=genai            # Specify the partition
#SBATCH --gres=gpu:1                 # Request 1 GPU

eval "$(pixi shell-hook -s bash)"


# Start the Ollama server in the background and log output
$SUMO_NLP_HOME/src/utils/start_ollama.sh

# Give some time for the server to start
sleep 5

# Optionally check if the server is running (example for a process check)
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Ollama server failed to start. Exiting..." >&2
    exit 1
fi

# Run the parallel Python script and redirect output
pixi run python -u weird_detection_for_training_format.py > weird_detection_output.txt