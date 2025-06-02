import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image
import os

# === Get the directory where the script is located ===
script_dir = os.path.dirname(os.path.abspath(__file__))

# === Paths (edited to be relative to the script location) ===
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"  # or v1-5
# Assuming 'trained_models' folder is in the same directory as your script
lora_path = os.path.join(script_dir, "Characters/Milo/trained_models/milo-20250522-011112/lora.safetensors")
embedding_path = os.path.join(script_dir, "Characters/Milo/trained_models/milo-20250522-011112/embeddings_converted.pti")
# It MUST match the token associated with your embeddings.pti file.
trigger_token = "<s0><s1>"
output_path = os.path.join(script_dir, "output.png") # Save output in the script's directory

# === Generation Parameters from UI (as per your images) ===
ui_negative_prompt = "Hyperrealistic, 4k, Photorealistic"
ui_width = 1024
ui_height = 1024
ui_num_inference_steps = 50
ui_guidance_scale = 7.5
ui_num_images_per_prompt = 1 # Corresponds to num_outputs in UI
ui_lora_scale = 0.9          # Corresponds to lora_scale in UI

# === Device selection ===
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# === Verify LoRA path ===
if not os.path.isfile(lora_path):
    print(f"Error: LoRA file not found at {lora_path}")
    print(f"Please ensure the path is correct and the file exists.")
    exit()
else:
    print(f"Found LoRA file at: {lora_path}")

# === Verify Embedding path ===
if not os.path.isfile(embedding_path):
    print(f"Warning: Embedding file not found at {embedding_path}.")
    print(f"The trigger token '{trigger_token}' may not work as intended if it relies on this embedding.")
    # Depending on your needs, you might want to exit() here if the embedding is essential
else:
    print(f"Found embedding file at: {embedding_path}")


# === Load the base pipeline ===
print("Loading base pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None,
)

# === Set the Scheduler (K_EULER from UI) ===
print(f"Setting scheduler to EulerDiscreteScheduler (K_EULER)...")
scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler

pipe = pipe.to(device)
print("Base pipeline and scheduler loaded and moved to device.")

# === Handling Textual Inversion Embeddings (.pti file) ===
# Load this BEFORE LoRA if your LoRA was trained with awareness of these TI tokens.
if os.path.isfile(embedding_path):
    print(f"Loading textual inversion embedding from: {embedding_path} for token '{trigger_token}'")
    try:
        # The load_textual_inversion function should handle .pti files.
        # It modifies the tokenizer and text encoder of the pipeline.
        pipe.load_textual_inversion(embedding_path, token=trigger_token)
        print(f"Textual inversion for '{trigger_token}' loaded successfully.")
    except Exception as e:
        print(f"Error loading textual inversion: {e}")
        print("Please ensure the embedding_path is correct and the file is a valid textual inversion embedding.")
        print(f"CRITICAL: Also ensure the 'trigger_token' variable ('{trigger_token}') in the script perfectly matches the token from your Replicate JSON metadata for this embedding.")
else:
    # This case is already handled by the verification step above,
    # but we keep this structure in case you want different logic here.
    print(f"Skipping textual inversion loading as file not found at: {embedding_path}")


# === Load LoRA weights ===
print(f"Loading LoRA weights from {lora_path}...")
pipe.load_lora_weights(lora_path)
print("LoRA weights loaded.")


# === Run the pipeline with UI parameters ===
# Ensure your prompt uses the exact `trigger_token` as defined and verified.
prompt = f"{trigger_token} milo holding a lolipop wearing a backpack"

print("Generating image with specified parameters...")
image = pipe(
    prompt=prompt,
    negative_prompt=ui_negative_prompt,
    width=ui_width,
    height=ui_height,
    num_inference_steps=ui_num_inference_steps,
    guidance_scale=ui_guidance_scale,
    num_images_per_prompt=ui_num_images_per_prompt,
    cross_attention_kwargs={"scale": ui_lora_scale} # Scales the LoRA effect
).images[0]
print("Image generated.")

# Save the image
image.save(output_path)
print(f"Saved image to {output_path}")