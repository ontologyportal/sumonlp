#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../load_configs.sh
echo "Installing requirements..."
conda config --set quiet true
conda install pip
pip install -U -q -r requirements.txt