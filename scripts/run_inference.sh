#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

LORA_MODEL="dreambooth_lora_ckpt"
STRENGTH=0.4
PROMPT="A photo of <vobj> wearing a suit and tie"

python src/two_stage_infer.py \
--lora_path "$LORA_MODEL/" \
--strength $STRENGTH \
--prompt "$PROMPT" \
--infer_steps 30 \
--save_path "output" \
--controlnet_conditioning_scale 0.7 \
--guidance_scale 6 \
--ref_style_img "style_refs/vangogh/The Church at Auvers.jpg \
--gpu 1