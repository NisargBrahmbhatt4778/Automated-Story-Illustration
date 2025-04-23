import torch
from diffusers import DiffusionPipeline

#  Update this to match the **exact path** where you've stored the SDXL model
MODEL_PATH = "models/sdxl-base-1.0"

#  Load SDXL Base Model (Optimized for Mac M3)
pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # Faster computation
    variant="fp16"
)
pipe.to("mps")  #  Enable Apple MPS (Metal Performance Shaders)

def generate_sdxl_image(prompt, filename="storybook_character.png"):
    """
    Generates an image using Stable Diffusion XL (SDXL) from the given prompt.
    """
    # Define negative prompts to avoid unwanted styles
    negative_prompt = "realistic, photo, 3D, horror, dark lighting, detailed, complex, photorealistic, hyperrealistic, cinematic, dramatic lighting"
    
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,  # More steps = better quality
        guidance_scale=7.5,  # Controls prompt adherence
        width=1024,  # SDXL is optimized for 1024x1024 resolution
        height=1024
    ).images[0]

    image.save(filename)
    print(f"Image saved as {filename}")


if __name__ == "__main__":
    #  Read the **LLaMA-generated** prompt from file
    with open("stable_diffusion_prompt.txt", "r") as file:
        stable_diffusion_prompt = file.read().strip()

    #  Generate the SDXL Image
    generate_sdxl_image(stable_diffusion_prompt, "storybook_character.png")
