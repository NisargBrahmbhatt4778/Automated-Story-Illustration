"""
Test script for the image cutting functionality.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cut_an_image import process_character_images

def test_image_cutting():
    """
    Test the image cutting functionality for a specific character.
    """
    print("\n=== Testing Image Cutting ===")
    
    # Get character name from user
    character_name = input("\nEnter the character name to test: ").strip()
    if not character_name:
        print("Error: Character name cannot be empty!")
        return
    
    # Construct the path to the uncut_images directory
    uncut_dir = Path('characters') / character_name / 'uncut_images'
    
    if not uncut_dir.exists():
        print(f"Error: Could not find uncut_images directory at {uncut_dir}")
        return
    
    print(f"\nFound uncut_images directory at: {uncut_dir}")
    print("Starting image processing...")
    
    # Process the images
    cut_dir = process_character_images(character_name, uncut_dir)
    
    if cut_dir:
        print("\nTest completed successfully!")
        print(f"Cut images are saved in: {cut_dir}")
        
        # Show some statistics
        cut_files = list(cut_dir.glob("*.png"))
        print(f"\nStatistics:")
        print(f"- Number of cut images: {len(cut_files)}")
        
        # Show first cut image dimensions if available
        if cut_files:
            from PIL import Image
            with Image.open(cut_files[0]) as img:
                print(f"- Cut image dimensions: {img.size}")
    else:
        print("\nTest failed!")

if __name__ == "__main__":
    try:
        test_image_cutting()
    except KeyboardInterrupt:
        print("\n\nImage cutting test interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1) 