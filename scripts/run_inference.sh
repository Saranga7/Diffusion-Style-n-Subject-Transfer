#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

LORA_MODEL="dreambooth_lora_ckpt_dog"
STRENGTH=0.5
PROMPT="A photo of <vobj> dog in a desert"

python src/two_stage_infer.py \
--lora_path "$LORA_MODEL/" \
--strength $STRENGTH \
--prompt "$PROMPT" \
--infer_steps 20 \
--controlnet_conditioning_scale 0.3 \
--guidance_scale 4 \
--ref_style_img "style_refs/rayonism/green-forest.jpg" \
--gpu 1