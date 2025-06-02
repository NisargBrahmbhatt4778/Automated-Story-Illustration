"""
This module handles the generation of new images using the trained model via Replicate API.
"""

import replicate
import requests
from pathlib import Path
from generation_history import GenerationHistory
from datetime import datetime

# Default parameters for image generation
DEFAULT_PARAMS = {
    "width": 1024,
    "height": 1024,
    "refine": "no_refiner",
    "scheduler": "K_EULER",
    "lora_scale": 0.9,
    "num_outputs": 1,
    "guidance_scale": 7.5,
    "apply_watermark": True,
    "high_noise_frac": 0.4,
    "negative_prompt": "Hyperrealistic, 4k, Photorealistic",
    "prompt_strength": 0.8,
    "num_inference_steps": 50
}

def get_custom_params():
    """
    Get custom parameters from user input.
    
    Returns:
        dict: Dictionary of custom parameters or None if user chooses defaults
    """
    print("\nWould you like to customize the generation parameters? (y/n)")
    choice = input("Enter your choice: ").strip().lower()
    
    if choice != 'y':
        return None
    
    custom_params = DEFAULT_PARAMS.copy()
    
    print("\n=== Customize Parameters ===")
    print("Enter new values for parameters you want to change.")
    print("Press Enter to keep the default value.")
    
    # Width and Height
    try:
        width = input(f"Width (default: {DEFAULT_PARAMS['width']}): ").strip()
        if width:
            custom_params['width'] = int(width)
        
        height = input(f"Height (default: {DEFAULT_PARAMS['height']}): ").strip()
        if height:
            custom_params['height'] = int(height)
    except ValueError:
        print("Invalid input for width/height. Using default values.")
    
    # Scheduler
    print("\nAvailable schedulers: K_EULER, DPM_SOLVER, DDIM, PNDM")
    scheduler = input(f"Scheduler (default: {DEFAULT_PARAMS['scheduler']}): ").strip()
    if scheduler:
        custom_params['scheduler'] = scheduler
    
    # LoRA Scale
    try:
        lora_scale = input(f"LoRA Scale (0.0-1.0, default: {DEFAULT_PARAMS['lora_scale']}): ").strip()
        if lora_scale:
            scale = float(lora_scale)
            if 0 <= scale <= 1:
                custom_params['lora_scale'] = scale
            else:
                print("LoRA scale must be between 0 and 1. Using default value.")
    except ValueError:
        print("Invalid input for LoRA scale. Using default value.")
    
    # Number of outputs
    try:
        num_outputs = input(f"Number of outputs (1-4, default: {DEFAULT_PARAMS['num_outputs']}): ").strip()
        if num_outputs:
            num = int(num_outputs)
            if 1 <= num <= 4:
                custom_params['num_outputs'] = num
            else:
                print("Number of outputs must be between 1 and 4. Using default value.")
    except ValueError:
        print("Invalid input for number of outputs. Using default value.")
    
    # Guidance Scale
    try:
        guidance_scale = input(f"Guidance Scale (1.0-20.0, default: {DEFAULT_PARAMS['guidance_scale']}): ").strip()
        if guidance_scale:
            scale = float(guidance_scale)
            if 1 <= scale <= 20:
                custom_params['guidance_scale'] = scale
            else:
                print("Guidance scale must be between 1 and 20. Using default value.")
    except ValueError:
        print("Invalid input for guidance scale. Using default value.")
    
    # Apply Watermark
    watermark = input(f"Apply Watermark (y/n, default: {'y' if DEFAULT_PARAMS['apply_watermark'] else 'n'}): ").strip().lower()
    if watermark:
        custom_params['apply_watermark'] = watermark == 'y'
    
    # High Noise Fraction
    try:
        noise_frac = input(f"High Noise Fraction (0.0-1.0, default: {DEFAULT_PARAMS['high_noise_frac']}): ").strip()
        if noise_frac:
            frac = float(noise_frac)
            if 0 <= frac <= 1:
                custom_params['high_noise_frac'] = frac
            else:
                print("High noise fraction must be between 0 and 1. Using default value.")
    except ValueError:
        print("Invalid input for high noise fraction. Using default value.")
    
    # Negative Prompt
    neg_prompt = input(f"Negative Prompt (default: {DEFAULT_PARAMS['negative_prompt']}): ").strip()
    if neg_prompt:
        custom_params['negative_prompt'] = neg_prompt
    
    # Prompt Strength
    try:
        prompt_strength = input(f"Prompt Strength (0.0-1.0, default: {DEFAULT_PARAMS['prompt_strength']}): ").strip()
        if prompt_strength:
            strength = float(prompt_strength)
            if 0 <= strength <= 1:
                custom_params['prompt_strength'] = strength
            else:
                print("Prompt strength must be between 0 and 1. Using default value.")
    except ValueError:
        print("Invalid input for prompt strength. Using default value.")
    
    # Number of Inference Steps
    try:
        steps = input(f"Number of Inference Steps (10-100, default: {DEFAULT_PARAMS['num_inference_steps']}): ").strip()
        if steps:
            num_steps = int(steps)
            if 10 <= num_steps <= 100:
                custom_params['num_inference_steps'] = num_steps
            else:
                print("Number of inference steps must be between 10 and 100. Using default value.")
    except ValueError:
        print("Invalid input for number of inference steps. Using default value.")
    
    return custom_params

def download_image(url, save_path):
    """
    Download an image from a URL and save it to the specified path.
    
    Args:
        url (str): URL of the image to download
        save_path (Path): Path where the image should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return False

def generate_new_image(model_id, character_name, history):
    """
    Generate a new image using the trained model via Replicate API.
    
    Args:
        model_id (str): The ID of the trained model
        character_name (str): Name of the character
        history (GenerationHistory): Instance of GenerationHistory for storing results
    
    Returns:
        list: List of generated image URLs or None if failed
    """
    print("\n=== Generate New Image ===")
    print("Please provide a prompt describing what you want the character to do or how you want them to appear.")
    print("The character's name will be automatically added to the prompt.")
    
    user_prompt = input("\nEnter your prompt: ").strip()
    while not user_prompt:
        print("Prompt cannot be empty!")
        user_prompt = input("Enter your prompt: ").strip()
    
    # Construct the full prompt with the character's name
    full_prompt = f"Illustration of {character_name} {user_prompt}"
    
    # Get custom parameters if user wants to customize
    params = get_custom_params() or DEFAULT_PARAMS
    
    try:
        output = replicate.run(
            model_id,
            input={
                "prompt": full_prompt,
                **params
            }
        )
        print("\nImage generated successfully!")
        
        # Save the generation to history
        if output:
            generation_id = history.save_generation(
                character_name=character_name,
                model_id=model_id,
                prompt=full_prompt,
                parameters=params,
                image_urls=output
            )
            print(f"\nGeneration saved with ID: {generation_id}")
            
            # Download the images
            base_dir = Path("generated_images") / character_name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, url in enumerate(output):
                filename = f"gen_{timestamp}_{i+1}.png"
                image_path = base_dir / filename
                if download_image(url, image_path):
                    print(f"Image saved to: {image_path}")
                else:
                    print(f"Failed to save image: {url}")
        
        return output
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

def generate_images_loop(model_id, character_name):
    """
    Run a loop to generate multiple images based on user input.
    
    Args:
        model_id (str): The ID of the trained model
        character_name (str): Name of the character
    """
    # Initialize history
    history = GenerationHistory()
    
    while True:
        print("\n=== Image Generation Menu ===")
        print("1. Generate new image")
        print("2. View generation history")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            generated_images = generate_new_image(model_id, character_name, history)
            if not generated_images:
                print("Error: Failed to generate images")
                continue
                
            print("\nGenerated image URLs:")
            for url in generated_images:
                print(url)
        
        elif choice == "2":
            print("\n=== Generation History ===")
            generations = history.get_generations(character_name)
            
            if not generations:
                print("No generations found for this character.")
                continue
            
            for gen in generations:
                print(f"\nGeneration ID: {gen['id']}")
                print(f"Created at: {gen['created_at']}")
                print(f"Prompt: {gen['prompt']}")
                print("Parameters:")
                for key, value in gen['parameters'].items():
                    print(f"  {key}: {value}")
                print("Images:")
                for path in gen['image_paths']:
                    print(f"  {path}")
                
                delete = input("\nWould you like to delete this generation? (y/n): ").strip().lower()
                if delete == 'y':
                    if history.delete_generation(gen['id']):
                        print("Generation deleted successfully.")
                    else:
                        print("Failed to delete generation.")
        
        elif choice == "3":
            break
        
        else:
            print("Invalid choice. Please try again.")
    
    print("\nThank you for using the character generator!") 