#!/bin/bash

# python -m nmt.nmt \
#   --encoder_type=bi \
#   --attention=scaled_luong \
#   --num_units=512 \
#   --num_gpus=2 \
#   --batch_size=768 \
#   --src=nat --tgt=sum1 \
#   --vocab_prefix=/home/mptp/big2/adamp/v1/vocab \
#   --train_prefix=/home/mptp/big2/adamp/v1/train \
#   --dev_prefix=/home/mptp/big2/adamp/v1/dev \
#   --test_prefix=/home/mptp/big2/adamp/v1/test \
#   --out_dir=/home/mptp/nmt1/tg1/model92 \
#   --num_train_steps=18000 \
#   --steps_per_stats=2000 \
#   --steps_per_external_eval=2000 \
#   --num_layers=2 \
#   --dropout=0.2 \
#   --metrics=bleu > /home/mptp/nmt1/tg1/train.log12kb92
#  --beam_width=10 \
#  --num_translations_per_input=10 > /home/mptp/nmt1/tg1/train.log

# cat /home/qwang/nmt/miz2pref/data/test.miz /home/qwang/nmt/miz2pref/data/dev.miz > /home/qwang/nmt/miz2pref/data/pump.miz
# cat /home/qwang/nmt/miz2pref/data/test.pref /home/qwang/nmt/miz2pref/data/dev.pref > /home/qwang/nmt/miz2pref/data/pump-compare.pref

# python -m nmt.nmt \
#   --beam_width=10 \
#   --num_translations_per_input=10 \
#   --out_dir=/home/qwang/nmt/miz2pref/model \
#   --inference_input_file=/home/qwang/nmt/miz2pref/data/pump.miz \
#   --inference_output_file=/home/qwang/nmt/miz2pref/data/pump.pref > /home/qwang/nmt/miz2pref/infer.log

python -m nmt.nmt      --out_dir=/home/mptp/nmt1/tg1/model92   --inference_input_file=/home/mptp/big2/adamp/v1/bigtest.nat  --inference_output_file=/home/mptp/big2/adamp/v1/bigtest.sum1infer
