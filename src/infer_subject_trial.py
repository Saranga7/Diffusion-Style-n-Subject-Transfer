from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from peft import PeftModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
pipeline.to(device)

# replace tokenizer with fine-tuned tokenizer
pipeline.tokenizer = CLIPTokenizer.from_pretrained("./dreambooth_lora_ckpt")
print(len(pipeline.tokenizer))
# replace text encoder with base model then load text LoRA adapter
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
text_encoder.resize_token_embeddings(len(pipeline.tokenizer))

pipeline.text_encoder = PeftModel.from_pretrained(text_encoder, "./dreambooth_lora_ckpt/text_lora_epoch1")

# load unet LoRA
pipeline.unet = PeftModel.from_pretrained(pipeline.unet, 
                                          "./dreambooth_lora_ckpt/unet_lora_epoch1")


generator = torch.Generator("cuda").manual_seed(777)

# now you can prompt using the new token:
prompt = "A photo of <vobj> in space"
out = pipeline(prompt = prompt, 
            #    image=..., 
               num_inference_steps = 30, 
               strength = 0.5,
               generator = generator).images[0]

out.save("output.png")



