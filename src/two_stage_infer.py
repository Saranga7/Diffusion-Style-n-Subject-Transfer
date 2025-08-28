import os
import torch
from helper import load_pipeline
from controlnet_aux import CannyDetector
from diffusers.utils import load_image, make_image_grid
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type = str, required = True)
    parser.add_argument('--prompt', type = str, required = True)
    parser.add_argument('--ref_style_img', type = str, required = True)

    parser.add_argument('--strength', type = float, default = 0.5)
    parser.add_argument('--infer_steps', type = int, default = 30)
    # parser.add_argument('--save_path', type = str, default='out')
    parser.add_argument('--controlnet_conditioning_scale', type = float, default = 0.7)
    parser.add_argument('--guidance_scale', type = float, default = 6)
    parser.add_argument('--gpu', type = int, default = 1)

    # parser.add_argument('--seed', type = int, default = 7)
    args = vars(parser.parse_args())

    device = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"

    # load pipeline with subject LoRAs

    unet_lora_path = os.path.join(args['lora_path'], "unet_lora_epoch7")
    text_lora_path = os.path.join(args['lora_path'], "text_lora_epoch7")

    pipeline = load_pipeline(
        pipeline_type='txt2img',
        subject_ckpt_dir = args['lora_path'],
        lora_unet_dir = unet_lora_path,
        lora_text_dir = text_lora_path,
        device = device
    )

    # ---------- Stage 1: txt->img (prompt to image)

    # prompt = "A photo of <vobj> in a natural setting, waterfall in the background"
    stage_a = pipeline(prompt = args['prompt'], 
                num_inference_steps = args['infer_steps'], 
                guidance_scale = args['guidance_scale']).images[0]
    stage_a.save("./stageA_img.png")


    # --------- Stage 2 : img->img (add style to subject-image)
    pipeline = load_pipeline(
        pipeline_type='img2img',
        device=device
    )

    canny = CannyDetector()
    canny_img = canny(stage_a, detect_resolution=768, image_resolution=768)

    ip_adap_img = load_image(args['ref_style_img']).resize((768, 768))

    images = pipeline(prompt = "A photo", 
                negative_prompt = "low quality",
                height = 768, 
                width = 768,
                ip_adapter_image = ip_adap_img,
                image = canny_img,
                guidance_scale = args['guidance_scale'],
                controlnet_conditioning_scale = args['controlnet_conditioning_scale'],
                num_inference_steps = args['infer_steps'],
                num_images_per_prompt = 3).images

    images = [(stage_a.resize((768, 768)))] + images

    out = make_image_grid(images, rows = 1, cols = 4)

    out.save("./stageB_final_image.png")
    print("Saved final image")



if __name__ == "__main__":
    main()
    