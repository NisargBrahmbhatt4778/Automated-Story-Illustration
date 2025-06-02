"""
This module provides functionality to test trained models by generating images.
It allows users to select a trained model and generate test images with different prompts.
"""

import os
import json
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load environment variables from .env file
load_dotenv()

# Set Replicate API token
os.environ["REPLICATE_IMG_GENERATION_API_TOKEN"] = os.getenv("REPLICATE_IMG_GENERATION_API_TOKEN")
if not os.getenv("REPLICATE_IMG_GENERATION_API_TOKEN"):
    print("Error: REPLICATE_IMG_GENERATION_API_TOKEN not found in .env file")
    sys.exit(1)

from image_generator import generate_images_loop
from generation_history import GenerationHistory

def get_trained_models():
    """
    Get a list of all trained models from the Characters directory structure.
    
    Returns:
        list: List of dictionaries containing model information
    """
    characters_dir = Path("Characters")
    if not characters_dir.exists():
        print("No Characters directory found. Please train a model first.")
        return []
    
    models = []
    # Iterate through character directories
    for char_dir in characters_dir.iterdir():
        if not char_dir.is_dir():
            continue
            
        # Look for model info file directly in character directory
        info_file = char_dir / "model_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    model_info = json.load(f)
                    model_info['directory'] = str(char_dir)
                    model_info['character_name'] = char_dir.name  # Ensure character name is set
                    models.append(model_info)
            except json.JSONDecodeError:
                print(f"Error reading model info for {char_dir.name}")
                continue
    
    return models

def display_models(models):
    """
    Display available models in a formatted way.
    
    Args:
        models (list): List of model information dictionaries
    """
    if not models:
        print("No trained models found.")
        return
    
    print("\n=== Available Trained Models ===")
    for i, model in enumerate(models, 1):
        print(f"\n{i}. Model: {model.get('destination_model', 'Unnamed Model')}")
        print(f"   Character: {model.get('character_name', 'Unknown')}")
        print(f"   Trained on: {model.get('trained_at', 'Unknown date')}")
        print(f"   Model ID: {model.get('model_id', {}).get('version', 'Unknown')}")
        print(f"   Directory: {model.get('directory', 'Unknown')}")

def select_model():
    """
    Let the user select a trained model.
    
    Returns:
        tuple: (model_id, character_name) or (None, None) if no model selected
    """
    models = get_trained_models()
    if not models:
        return None, None
    
    display_models(models)
    
    while True:
        try:
            choice = input("\nEnter the number of the model to use (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None, None
            
            index = int(choice) - 1
            if 0 <= index < len(models):
                selected_model = models[index]
                return selected_model['model_id']['version'], selected_model['character_name']
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    """
    Main function to run the model testing interface.
    """
    print("=== Model Testing Interface ===")
    print("This interface allows you to test trained models by generating new images.")
    
    while True:
        print("\n=== Main Menu ===")
        print("1. Select a model and generate images")
        print("2. View generation history")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            model_id, character_name = select_model()
            if model_id and character_name:
                print(f"\nSelected model for character: {character_name}")
                generate_images_loop(model_id, character_name)
        
        elif choice == "2":
            history = GenerationHistory()
            print("\n=== Generation History ===")
            print("1. View all generations")
            print("2. View generations for a specific character")
            
            view_choice = input("\nEnter your choice (1-2): ").strip()
            
            if view_choice == "1":
                generations = history.get_generations()
            elif view_choice == "2":
                character = input("Enter character name: ").strip()
                generations = history.get_generations(character_name=character)
            else:
                print("Invalid choice.")
                continue
            
            if not generations:
                print("No generations found.")
                continue
            
            for gen in generations:
                print(f"\nGeneration ID: {gen['id']}")
                print(f"Character: {gen['character_name']}")
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
            print("\nThank you for using the model testing interface!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 