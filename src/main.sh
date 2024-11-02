#!/bin/bash

echo "Starting SUMO Language to Logic conversion ..."


bash metaphor_handling/entry_point.sh
bash sentence_simplification/entry_point.sh
bash oov_handling/entry_point.sh
bash l2l/entry_point.sh


