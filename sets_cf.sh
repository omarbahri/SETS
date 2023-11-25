#!/bin/bash

#this file runs SETS on the solar flare dataset as described in the paper
#feel free to experiment

dataset_name='sf'
st_contract='220'
max_sh_len='50'

python sets/st.py $dataset_name $st_contract $max_sh_len
python sets/class_shapelets.py $dataset_name $st_contract $max_sh_len
python sets/sets.py $dataset_name $st_contract $max_sh_len