"""
Test file for running just the SDXL model training step.
"""

import os
from pathlib import Path
from src.model_trainer import train_sdxl_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_character_name():
    """Get the character name from user input."""
    while True:
        character_name = input("\nEnter the character name: ").strip()
        if not character_name:
            print("Character name cannot be empty!")
            continue
        
        # Check if the character directory exists
        char_dir = Path(f"characters/{character_name}")
        if not char_dir.exists():
            print(f"Error: No directory found for character '{character_name}'")
            print("Please make sure the character directory exists at:", char_dir)
            continue
            
        # Check if cut_images directory exists
        cut_dir = char_dir / "cut_images"
        if not cut_dir.exists():
            print(f"Error: No cut_images directory found for character '{character_name}'")
            print("Please make sure the cut_images directory exists at:", cut_dir)
            continue
            
        # Check if there are any PNG images
        if not any(cut_dir.glob("*.png")):
            print(f"Error: No PNG images found in {cut_dir}")
            print("Please add some PNG images for training.")
            continue
            
        return character_name

def test_training():
    # Get character name from user
    character_name = get_character_name()
    
    # Path to the directory containing training images
    training_images_dir = Path(f"characters/{character_name}/cut_images")
    
    # Check for Replicate API token
    if not os.getenv('REPLICATE_API_TOKEN'):
        print("Error: REPLICATE_API_TOKEN not found in .env file")
        print("Please create a .env file in the project root with your Replicate API token:")
        print("REPLICATE_API_TOKEN=your_token_here")
        return
    
    print("\n=== Starting Test Training ===")
    print(f"Character Name: {character_name}")
    print(f"Training Images Directory: {training_images_dir}")
    print(f"Number of training images: {len(list(training_images_dir.glob('*.png')))}")
    
    # Run the training (no description needed for test)
    model_id = train_sdxl_model(character_name, training_images_dir, "")
    
    if model_id:
        print("\nTraining completed successfully!")
        print(f"Model ID: {model_id}")
        print(f"Model information saved to: characters/{character_name}/model_info.json")
    else:
        print("\nTraining failed!")

if __name__ == "__main__":
    test_training() 