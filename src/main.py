"""
This is the main file for the storybook character generator.

It will run the differnt stages of the charecter gen pipleine. The stagesa are as follows:

1. Get a character name + Description description from the user
    1a. Filter the charecter description to remove the charecter name and make sure everything in the description only 
2. Generate a character sheet and an action sheet for the character (manual step)
    2a. Using the name and description, create a prompt to give to ChatGPT to generate the character sheet and action sheet.
3. The user places all the sheets in a folder names the character name
4. Run some sort of code that then cuts up the charecter sheet into separate charecter images
5. Upscale the cut images using Real-ESRGAN (Anime6B) for better quality - This now happens before the sheet is cut
6. Call the API to train the SDXL model on the charecter images.
7. Using the trained model, ask it to generate another image using the original character description.
8. Iterative step for the user to keep asking for new images of the same charecter.


Step 6 to be expanded and made much better in the future.

"""

from automated_sheet_handler import handle_character_sheets
from cut_an_image import process_character_images
from upscaler import upscale_character_images
from model_trainer import train_sdxl_model
from image_generator import generate_images_loop
from pipeline_logger import init_pipeline_logger, get_pipeline_logger, cleanup_pipeline_logger
import replicate

# Step 1 - Get Character Name + Description
def get_character_info():
    print("\n=== Character Information Input ===")
    character_name = input("Enter the character's name: ").strip()
    while not character_name:
        print("Character name cannot be empty!")
        character_name = input("Enter the character's name: ").strip()
    
    print("\nPlease provide a detailed description of the character.")
    print("Include details like:")
    print("- Physical appearance (height, build, hair color, etc.)")
    print("- Clothing and accessories")
    print("- Notable features or characteristics")
    print("- Any specific style preferences")
    
    character_description = input("\nEnter character description: ").strip()
    while not character_description:
        print("Character description cannot be empty!")
        character_description = input("Enter character description: ").strip()
    
    return character_name, character_description

def upscale_and_process_images(character_name, uncut_dir):
    """
    First upscale the full sheets, then process them by cutting into sections.
    
    Args:
        character_name (str): Name of the character
        uncut_dir (Path): Path to the directory containing uncut images
        
    Returns:
        tuple: (Path to cut images directory, Path to upscaled images directory) or (None, None) if failed
    """
    pipeline_logger = get_pipeline_logger()
    
    pipeline_logger.log_info("  → Starting image upscaling process...")
    
    # Step 1: Upscale the full sheets
    upscaled_dir = upscale_character_images(character_name, uncut_dir)
    if not upscaled_dir:
        pipeline_logger.log_error("  ✗ Failed to upscale character sheets")
        return None, None
    
    pipeline_logger.log_info(f"  ✓ Character sheets upscaled successfully: {upscaled_dir}")
    
    # Step 2: Process (cut) the upscaled images
    pipeline_logger.log_info("  → Starting image cutting process...")
    cut_dir = process_character_images(character_name, upscaled_dir)
    if not cut_dir:
        pipeline_logger.log_error("  ✗ Failed to process upscaled images")
        return None, upscaled_dir
    
    pipeline_logger.log_info(f"  ✓ Image processing completed successfully: {cut_dir}")
    return cut_dir, upscaled_dir

# Step 7 - Generate a new image using the trained model by making a call to the Replicate API
def generate_new_image(model_id, character_name):
    """
    Generate a new image using the trained model via Replicate API.
    
    Args:
        model_id (str): The ID of the trained model
        character_name (str): Name of the character
    
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
    
    try:
        output = replicate.run(
            model_id,
            input={
                "width": 1024,
                "height": 1024,
                "prompt": full_prompt,
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
        )
        print("\nImage generated successfully!")
        return output
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

if __name__ == "__main__":
    # Step 1: Get character information
    character_name, character_description = get_character_info()
    
    # Initialize pipeline logger
    pipeline_logger = init_pipeline_logger(character_name)
    pipeline_logger.log_pipeline_start(character_description)
    
    try:
        # Step 2: Handle character sheet generation
        pipeline_logger.log_stage_start(2, "Character Sheet Generation", "Generating character, action, and emotion sheets using OpenAI API")
        uncut_dir = handle_character_sheets(character_name, character_description)
        if not uncut_dir:
            pipeline_logger.log_stage_error(2, "Character Sheet Generation", "Failed to generate character sheets")
            exit(1)
        pipeline_logger.log_stage_success(2, "Character Sheet Generation", f"Sheets saved to: {uncut_dir}")
        
        # Step 3: Upscale sheets and process images
        pipeline_logger.log_stage_start(3, "Image Processing", "Upscaling character sheets and cutting them into individual images")
        cut_dir, upscaled_dir = upscale_and_process_images(character_name, uncut_dir)
        if not cut_dir:
            pipeline_logger.log_stage_error(3, "Image Processing", "Failed to process images")
            exit(1)
        pipeline_logger.log_stage_success(3, "Image Processing", f"Cut images saved to: {cut_dir}")
        
        # Step 4: Train SDXL model on character images
        pipeline_logger.log_stage_start(4, "Model Training", "Training SDXL model on character images")
        model_id = train_sdxl_model(character_name, cut_dir, character_description)
        if not model_id:
            pipeline_logger.log_stage_error(4, "Model Training", "Failed to train SDXL model")
            exit(1)
        pipeline_logger.log_stage_success(4, "Model Training", f"Model trained successfully. Model ID: {model_id}")
        
        # Step 5: Generate new images using trained model
        pipeline_logger.log_stage_start(5, "Image Generation Loop", "Interactive image generation using trained model")
        generate_images_loop(model_id, character_name)
        pipeline_logger.log_stage_success(5, "Image Generation Loop", "User completed image generation session")
        
        # Pipeline completed successfully
        pipeline_logger.log_pipeline_complete(success=True, final_message=f"Character '{character_name}' pipeline completed successfully!")
        
    except Exception as e:
        pipeline_logger.log_error(f"Unexpected error in pipeline: {str(e)}")
        pipeline_logger.log_pipeline_complete(success=False, final_message=f"Pipeline failed with error: {str(e)}")
        exit(1)
    
    finally:
        # Clean up logger
        cleanup_pipeline_logger()
