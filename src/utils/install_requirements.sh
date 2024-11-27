#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../config_paths.sh
echo "Installing requirements..."
conda config --set quiet true
conda install pip
pip install -U -q -r requirements.txt