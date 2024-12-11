#!/bin/bash
echo "Starting Language to Logic translation ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
# source ../config_paths.sh

python reference/inference.py "$MODEL_HOME"

sed -i 's/? / ?/g' output_l2l.txt

echo "Finished Language to Logic translation ..."