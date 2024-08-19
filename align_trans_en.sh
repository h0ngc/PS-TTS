#!/bin/bash

txt_fpath="$1"

python make_ensub.py -i "$txt_fpath" -o "./english_trans.txt"
