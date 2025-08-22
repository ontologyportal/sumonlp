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
python -u src/inference.py "$SUMO_NLP_HOME/L2L_model/$MODEL_L2L/model.ckpt"



cd "$parent_path" || exit

if [ -f "$SUMO_NLP_HOME/L2L_model/$MODEL_L2L/combined-log.map" ]; then
  cp output_l2l.txt output_l2l_scrambled.txt
  echo "Scrambled translation: "
  cat output_l2l_scrambled.txt
  python semantic_separator.py --map "$SUMO_NLP_HOME/L2L_model/$MODEL_L2L/combined-log.map" unscramble output_l2l_scrambled.txt output_l2l.txt
else
    sed -i 's/? / ?/g' output_l2l.txt
    sed -i 's/?/ ?/g' output_l2l.txt
fi


echo "Finished Language to Logic translation ..."
