from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime

model_id = "mann-e/mann-e_4_rev-1-3"
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
pipe = pipe.to("cpu")


def dummy(images, **kwargs):
    return images, False


pipe.safety_checker = dummy

prompt = "elden ring style, a beautiful painting of a castle in a beautiful mountain landscape, fantasy, gloomy, blue hour, trending on artstation, concept art, digital painting, art by midjourney"
negative_prompt = "low quality, blurry"
width = 512
height = 512

print(f'prompt is {prompt}')
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    width=width,
    height=height,
    guidance_scale=10).images[0]

now = datetime.now()
image.save(f'images/{now.strftime("%Y%m%d-%H%M%S")}.png')
