from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

model_id = "mann-e/mann-e_4_rev-1-3"
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
pipe = pipe.to("cpu")

def dummy(images, **kwargs): 
    return images, False 

pipe.safety_checker = dummy

prompt = "a historical city in an (((island))), view from the sea, cyberpunk, flashing blue and orange neon lights, night, fantasy, highly detailed digital painting, trending on artstation, concept art, sharp focus, illustration, art by midjourney" 
negative_prompt = "low quality, blurry"
width = 512
height = 512 
