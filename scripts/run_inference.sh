#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

LORA_MODEL="dreambooth_lora_ckpt"
STRENGTH=0.5
PROMPT="A photo of <vobj> dog wearing a suit and tie"

python src/two_stage_infer.py \
--lora_path "$LORA_MODEL/" \
--strength $STRENGTH \
--prompt "$PROMPT" \
--infer_steps 20 \
--controlnet_conditioning_scale 0.7 \
--guidance_scale 6 \
--ref_style_img "style_refs/rayonism/electric-lamp.jpg" \
--gpu 1