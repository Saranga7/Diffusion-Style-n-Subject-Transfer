#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python src/train_dreambooth_lora.py \
  --instance_data_dir "Dali_images" \
  --class_data_dir "catsndogs/cats_and_dogs_small/validation/dogs" \
  --instance_token "<vobj>" \
  --class_word "dog" \
  --output_dir ./dreambooth_lora_ckpt \
  --num_epochs 3 \
  --batch_size 2 \
  --max_train_steps 2000 \
  --lr 1e-4 \
  --gpu 1
