#!/bin/bash

python -W ignore MetaPrompting.py \
  --mode MetaPT_MAML++ \
  --n_adapt_epochs 15 \
  --dataset huffpost \
  --out_dataset huffpost \
  --k_shot 1 \
  --prompt_template 0 \
  --no_train 0 \
  --no_eval 0

