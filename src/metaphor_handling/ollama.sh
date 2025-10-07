#!/bin/bash
# A shell wrapper that works reliably outside Python

# module purge
# module load lib/cuda/12.2    # these lines might be necessary for use on hamming
ollama serve > ./ollama_log.out 2>&1 &
