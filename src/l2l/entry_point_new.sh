#!/bin/bash
echo "Starting Language to Logic translation ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../load_configs.sh
# source ~/.bashrc

cd train || exit

eval "$(pixi shell-hook -s bash)"

python src/inference.py "$MODEL_CHECKPOINT"

sed -i 's/? / ?/g' output_l2l.txt
sed -i 's/?/ ?/g' output_l2l.txt

echo "Finished Language to Logic translation ..."

