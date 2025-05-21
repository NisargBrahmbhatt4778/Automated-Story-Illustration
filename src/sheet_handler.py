"""
This module handles the generation and upload of character sheets, action sheets, and emotion sheets.
"""

import os
import shutil
from pathlib import Path

def read_template(template_name):
    try:
        with open(f'z_GPT_Templates/{template_name}', 'r') as template_file:
            return template_file.read()
    except FileNotFoundError:
        print(f"Error: Could not find the template file '{template_name}'")
        return None

def display_prompt(prompt, title):
    print(f"\n=== {title} ===")
    print("Copy and paste the following prompt into ChatGPT to generate your sheet:")
    print("\n" + "="*50 + "\n")
    print(prompt)
    print("\n" + "="*50 + "\n")
    input("Press Enter after you have generated the image...")

def save_sheet(character_name, sheet_type, original_path):
    # Create the character's directory and uncut_images subdirectory if they don't exist
    char_dir = Path('Characters') / character_name
    uncut_dir = char_dir / 'uncut_images'
    char_dir.mkdir(parents=True, exist_ok=True)
    uncut_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the new filename
    new_name = f"{sheet_type}_Sheet_{character_name}"
    
    # Get the file extension from the original file
    file_ext = os.path.splitext(original_path)[1]
    
    # Create the new file path in the uncut_images directory
    new_file_path = uncut_dir / f"{new_name}{file_ext}"
    
    try:
        # Copy the file to the uncut_images directory with the new name
        shutil.copy2(original_path, new_file_path)
        print(f"Successfully saved {sheet_type} sheet as {new_name}{file_ext} in {uncut_dir}")
        return True
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False

def get_sheet_path(sheet_type):
    while True:
        print(f"\nPlease provide the path to the {sheet_type} sheet:")
        file_path = input("> ").strip()
        
        if not file_path:
            print("Please provide a valid file path!")
            continue
            
        if not os.path.exists(file_path):
            print("File not found! Please provide a valid path.")
            continue
            
        return file_path

def handle_character_sheets(character_name, character_description):
    """
    Handles the sequential generation and upload of character sheets.
    
    Args:
        character_name (str): The name of the character
        character_description (str): The description of the character
        
    Returns:
        Path: Path to the character's uncut_images directory if successful, None otherwise
    """
    # Step 1: Character Sheet
    char_template = read_template('GPT_Template_for_Sheet_Gen.txt')
    if char_template:
        char_prompt = char_template.replace('{Char_Description}', character_description)
        display_prompt(char_prompt, "Character Sheet Generation")
        
        # Get and save character sheet
        char_path = get_sheet_path("character")
        if not save_sheet(character_name, "Char", char_path):
            return None
    
    # Step 2: Action Sheet
    action_template = read_template('GPT_Template_for_Action_Sheet_Gen.txt')
    if action_template:
        action_prompt = action_template.replace('{Char_Description}', character_description)
        display_prompt(action_prompt, "Action Sheet Generation")
        
        # Get and save action sheet
        action_path = get_sheet_path("action")
        if not save_sheet(character_name, "Action", action_path):
            return None
    
    # Step 3: Emotion Sheet
    emotion_template = read_template('GPT_Template_for_Emotion_Sheet_Gen.txt')
    if emotion_template:
        emotion_prompt = emotion_template.replace('{Char_Description}', character_description)
        display_prompt(emotion_prompt, "Emotion Sheet Generation")
        
        # Get and save emotion sheet
        emotion_path = get_sheet_path("emotion")
        if not save_sheet(character_name, "Emotion", emotion_path):
            return None
    
    char_dir = Path('Characters') / character_name / 'uncut_images'
    print(f"\nAll sheets have been saved to: {char_dir}")
    return char_dir 