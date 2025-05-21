"""
This module handles the cutting of character sheets into individual images.
"""

from pathlib import Path
from PIL import Image
import os

def cut_character_sheet(image_path, output_dir, character_name):
    """
    Cuts a 1536x1024 character sheet into 6 equal sections.
    
    Args:
        image_path (Path): Path to the character sheet image
        output_dir (Path): Directory to save the cut images
        character_name (str): Name of the character for file naming
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Verify image dimensions
            if img.size != (1536, 1024):
                print(f"Warning: Character sheet dimensions are {img.size}, expected (1536, 1024)")
            
            # Calculate section dimensions
            section_width = img.width // 3  # 512
            section_height = img.height // 2  # 512
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Cut the image into 6 sections
            for row in range(2):
                for col in range(3):
                    # Calculate coordinates for this section
                    left = col * section_width
                    upper = row * section_height
                    right = left + section_width
                    lower = upper + section_height
                    
                    # Crop the section
                    section = img.crop((left, upper, right, lower))
                    
                    # Save the section
                    section_num = row * 3 + col + 1
                    output_path = output_dir / f"char_sheet_{section_num}.png"
                    section.save(output_path, "PNG")
                    print(f"Saved character section {section_num} to {output_path}")
                    
    except Exception as e:
        print(f"Error processing character sheet: {str(e)}")
        return False
    
    return True

def cut_action_sheet(image_path, output_dir, character_name):
    """
    Cuts a 1536x1024 action sheet into 6 equal sections.
    
    Args:
        image_path (Path): Path to the action sheet image
        output_dir (Path): Directory to save the cut images
        character_name (str): Name of the character for file naming
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Verify image dimensions
            if img.size != (1536, 1024):
                print(f"Warning: Action sheet dimensions are {img.size}, expected (1536, 1024)")
            
            # Calculate section dimensions
            section_width = img.width // 3  # 512
            section_height = img.height // 2  # 512
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Cut the image into 6 sections
            for row in range(2):
                for col in range(3):
                    # Calculate coordinates for this section
                    left = col * section_width
                    upper = row * section_height
                    right = left + section_width
                    lower = upper + section_height
                    
                    # Crop the section
                    section = img.crop((left, upper, right, lower))
                    
                    # Save the section
                    section_num = row * 3 + col + 1
                    output_path = output_dir / f"action_sheet_{section_num}.png"
                    section.save(output_path, "PNG")
                    print(f"Saved action section {section_num} to {output_path}")
                    
    except Exception as e:
        print(f"Error processing action sheet: {str(e)}")
        return False
    
    return True

def cut_emotion_sheet(image_path, output_dir, character_name):
    """
    Cuts a 1024x1024 emotion sheet into 9 equal sections.
    
    Args:
        image_path (Path): Path to the emotion sheet image
        output_dir (Path): Directory to save the cut images
        character_name (str): Name of the character for file naming
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Verify image dimensions
            if img.size != (1024, 1024):
                print(f"Warning: Emotion sheet dimensions are {img.size}, expected (1024, 1024)")
            
            # Calculate section dimensions
            section_size = img.width // 3  # 341
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Cut the image into 9 sections
            for row in range(3):
                for col in range(3):
                    # Calculate coordinates for this section
                    left = col * section_size
                    upper = row * section_size
                    right = left + section_size
                    lower = upper + section_size
                    
                    # Crop the section
                    section = img.crop((left, upper, right, lower))
                    
                    # Save the section
                    section_num = row * 3 + col + 1
                    output_path = output_dir / f"emotion_sheet_{section_num}.png"
                    section.save(output_path, "PNG")
                    print(f"Saved emotion section {section_num} to {output_path}")
                    
    except Exception as e:
        print(f"Error processing emotion sheet: {str(e)}")
        return False
    
    return True

def process_character_images(character_name, uncut_dir):
    """
    Processes all character images in the uncut_images directory.
    If upscaled images exist, they will be used instead.
    
    Args:
        character_name (str): Name of the character
        uncut_dir (Path): Path to the uncut_images directory
    """
    # Create cut_images directory
    cut_dir = uncut_dir.parent / 'cut_images'
    cut_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for upscaled images
    upscaled_dir = Path(f"characters/{character_name}/upscaled_images")
    use_upscaled = upscaled_dir.exists() and any(upscaled_dir.glob("*.png"))
    
    if use_upscaled:
        print("\nUsing upscaled images for processing...")
        input_dir = upscaled_dir
    else:
        print("\nUsing original images for processing...")
        input_dir = uncut_dir
    
    # Process character sheet
    char_sheet = input_dir / f"Char_Sheet_{character_name}.png"
    if char_sheet.exists():
        print("\nProcessing character sheet...")
        if not cut_character_sheet(char_sheet, cut_dir, character_name):
            return False
    else:
        print(f"Character sheet not found at {char_sheet}")
        return False
    
    # Process action sheet
    action_sheet = input_dir / f"Action_Sheet_{character_name}.png"
    if action_sheet.exists():
        print("\nProcessing action sheet...")
        if not cut_action_sheet(action_sheet, cut_dir, character_name):
            return False
    else:
        print(f"Action sheet not found at {action_sheet}")
        return False
    
    # Process emotion sheet
    emotion_sheet = input_dir / f"Emotion_Sheet_{character_name}.png"
    if emotion_sheet.exists():
        print("\nProcessing emotion sheet...")
        if not cut_emotion_sheet(emotion_sheet, cut_dir, character_name):
            return False
    else:
        print(f"Emotion sheet not found at {emotion_sheet}")
        return False
    
    print(f"\nAll images have been processed and saved to: {cut_dir}")
    return cut_dir 