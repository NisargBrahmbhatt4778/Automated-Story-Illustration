"""
Test script for the upscaling functionality of the character generator.
This script allows you to test the upscaling process on a single character's images.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from upscaler import setup_upscaler, upscale_character_images

def test_upscaling():
    """
    Test the upscaling functionality for a specific character.
    """
    print("\n=== Testing Image Upscaling ===")
    
    # Get character name from user
    character_name = input("\nEnter the character name to test upscaling: ")
    
    # Set up the upscaler
    upscaler = setup_upscaler()
    
    # Find all PNG files in the character's directory
    input_dir = Path(f"characters/{character_name}/uncut_images")
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist!")
        return
        
    png_files = list(input_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
        
    print(f"\nFound {len(png_files)} PNG files to upscale:")
    for file in png_files:
        print(f"- {file.name}")
        
    # Ask for confirmation
    proceed = input("\nProceed with upscaling? (y/n): ").lower()
    if proceed != 'y':
        print("Upscaling cancelled.")
        return
        
    print("\nStarting upscaling process...")
    
    # Upscale images
    output_dir = upscale_character_images(character_name, input_dir)
    
    if output_dir:
        print(f"\nUpscaled images saved to: {output_dir}")
    else:
        print("\nError: Upscaling failed!")

if __name__ == "__main__":
    try:
        test_upscaling()
    except KeyboardInterrupt:
        print("\n\nUpscaling test interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1) 