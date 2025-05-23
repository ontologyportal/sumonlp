#!/bin/bash
echo "Starting Language to Logic translation ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../load_configs.sh
# source ~/.bashrc

cd train || exit

eval "$(pixi shell-hook -s bash)"

# python src/inference.py "/home/angelos.toutsios.gr/data/Thesis_dev/L2L_model_training/out/2025-04-26_10-18-13-3rd/lightning_logs/version_0/checkpoints/epoch=3-val_loss=0.01010.ckpt"
python -u src/inference.py "/data/fsg/.sumonlp/model_generation/2025-05-12_flan_no_scramble/out/2025-05-14_22-03-56/lightning_logs/version_0/checkpoints/last.ckpt"

sed -i 's/? / ?/g' output_l2l.txt
sed -i 's/?/ ?/g' output_l2l.txt

echo "Finished Language to Logic translation ..."
