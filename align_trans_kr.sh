#!/bin/bash

txt_fpath="$1"

python make_krsub.py -i "$txt_fpath" -o "./korean_trans.txt"
