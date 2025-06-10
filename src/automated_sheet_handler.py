"""
Automated character sheet handler using OpenAI API.
This module automatically generates character sheets, action sheets, and emotion sheets using OpenAI's API.
"""

import os
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pipeline_logger import get_pipeline_logger

def read_template(template_name):
    """Read template file from z_GPT_Templates directory"""
    try:
        with open(f'z_GPT_Templates/{template_name}', 'r') as template_file:
            return template_file.read()
    except FileNotFoundError:
        print(f"Error: Could not find the template file '{template_name}'")
        return None

def image_to_data_url(image_path):
    """Convert image to base64 data URL for OpenAI API"""
    try:
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode('utf-8')
            data_url = f"data:image/png;base64,{base64_data}"
            return data_url
    except FileNotFoundError:
        print(f"Error: Could not find the image file '{image_path}'")
        return None

def save_generated_image(image_base64, file_path):
    """Save base64 encoded image to file"""
    try:
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        return True
    except Exception as e:
        print(f"Error saving image to {file_path}: {str(e)}")
        return False

def handle_character_sheets(character_name, character_description):
    """
    Automatically generates character sheets using OpenAI API.
    
    Args:
        character_name (str): The name of the character
        character_description (str): The description of the character
        
    Returns:
        Path: Path to the character's uncut_images directory if successful, None otherwise
    """
    # Get pipeline logger
    pipeline_logger = get_pipeline_logger()
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pipeline_logger.log_error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        return None
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
        pipeline_logger.log_info(" OpenAI client initialized successfully")
    except Exception as e:
        pipeline_logger.log_error(f"Failed to initialize OpenAI client: {str(e)}")
        return None
    
    # Create character directory structure
    char_dir = Path('Characters') / character_name
    uncut_dir = char_dir / 'uncut_images'
    char_dir.mkdir(parents=True, exist_ok=True)
    uncut_dir.mkdir(parents=True, exist_ok=True)
    pipeline_logger.log_info(f" Created directory structure: {uncut_dir}")
    
    # Read template files
    pipeline_logger.log_info("  → Loading template files...")
    
    char_template = read_template('GPT_Template_for_Sheet_Gen.txt')
    if not char_template:
        pipeline_logger.log_error("Failed to read character sheet template")
        return None
    
    action_template = read_template('GPT_Template_for_Action_Sheet_Gen.txt')
    if not action_template:
        pipeline_logger.log_error("Failed to read action sheet template")
        return None
    
    emotion_template = read_template('GPT_Template_for_Emotion_Sheet_Gen.txt')
    if not emotion_template:
        pipeline_logger.log_error("Failed to read emotion sheet template")
        return None
    
    pipeline_logger.log_info(" All templates loaded successfully")
    
    # Prepare templates with character description
    char_prompt = char_template.replace('{Char_Description}', character_description)
    action_prompt = action_template.replace('{Char_Description}', character_description)
    emotion_prompt = emotion_template.replace('{Char_Description}', character_description)
    
    # Load grid images for prompts
    pipeline_logger.log_info("  → Loading grid template images...")
    grid_data_url = image_to_data_url("z_GPT_Templates/grid_with_sections.png")
    square_data_url = image_to_data_url("z_GPT_Templates/grid_with_sections_square.png")
    
    if not grid_data_url or not square_data_url:
        pipeline_logger.log_error("Failed to load grid template images")
        return None
    
    pipeline_logger.log_info(" Grid templates loaded successfully")
    
    try:
        # Generate Character Sheet
        pipeline_logger.log_sheet_generation("Character", "start")
        pipeline_logger.log_api_call("OpenAI", "Character Sheet Generation", "start")
        
        char_response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": char_prompt},
                        {"type": "input_image", "image_url": grid_data_url}
                    ]
                }
            ],
            tools=[{"type": "image_generation", "size": "1536x1024"}]
        )
        
        char_image_calls = [
            output for output in char_response.output
            if output.type == "image_generation_call"
        ]
        
        if not char_image_calls:
            pipeline_logger.log_api_call("OpenAI", "Character Sheet Generation", "error", "No character sheet image generated")
            return None
        
        pipeline_logger.log_api_call("OpenAI", "Character Sheet Generation", "success")
        
        char_image_path = uncut_dir / f"Char_Sheet_{character_name}.png"
        pipeline_logger.log_file_operation("Saving Character Sheet", str(char_image_path), "start")
        
        if save_generated_image(char_image_calls[0].result, char_image_path):
            pipeline_logger.log_file_operation("Saving Character Sheet", str(char_image_path), "success")
            pipeline_logger.log_sheet_generation("Character", "success", str(char_image_path))
        else:
            pipeline_logger.log_file_operation("Saving Character Sheet", str(char_image_path), "error")
            return None
        
        # Generate Action Sheet
        pipeline_logger.log_sheet_generation("Action", "start")
        pipeline_logger.log_api_call("OpenAI", "Action Sheet Generation", "start")
        
        action_response = client.responses.create(
            model="gpt-4o",
            previous_response_id=char_response.id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": action_prompt},
                        {"type": "input_image", "image_url": grid_data_url}
                    ]
                }
            ],
            tools=[{"type": "image_generation", "size": "1536x1024"}]
        )
        
        action_image_calls = [
            output for output in action_response.output
            if output.type == "image_generation_call"
        ]
        
        if not action_image_calls:
            pipeline_logger.log_api_call("OpenAI", "Action Sheet Generation", "error", "No action sheet image generated")
            return None
        
        pipeline_logger.log_api_call("OpenAI", "Action Sheet Generation", "success")
        
        action_image_path = uncut_dir / f"Action_Sheet_{character_name}.png"
        pipeline_logger.log_file_operation("Saving Action Sheet", str(action_image_path), "start")
        
        if save_generated_image(action_image_calls[0].result, action_image_path):
            pipeline_logger.log_file_operation("Saving Action Sheet", str(action_image_path), "success")
            pipeline_logger.log_sheet_generation("Action", "success", str(action_image_path))
        else:
            pipeline_logger.log_file_operation("Saving Action Sheet", str(action_image_path), "error")
            return None
        
        # Generate Emotion Sheet
        pipeline_logger.log_sheet_generation("Emotion", "start")
        pipeline_logger.log_api_call("OpenAI", "Emotion Sheet Generation", "start")
        
        emotion_response = client.responses.create(
            model="gpt-4o",
            previous_response_id=action_response.id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": emotion_prompt},
                        {"type": "input_image", "image_url": square_data_url}
                    ]
                }
            ],
            tools=[{"type": "image_generation", "size": "1024x1024"}]
        )
        
        emotion_image_calls = [
            output for output in emotion_response.output
            if output.type == "image_generation_call"
        ]
        
        if not emotion_image_calls:
            pipeline_logger.log_api_call("OpenAI", "Emotion Sheet Generation", "error", "No emotion sheet image generated")
            return None
        
        pipeline_logger.log_api_call("OpenAI", "Emotion Sheet Generation", "success")
        
        emotion_image_path = uncut_dir / f"Emotion_Sheet_{character_name}.png"
        pipeline_logger.log_file_operation("Saving Emotion Sheet", str(emotion_image_path), "start")
        
        if save_generated_image(emotion_image_calls[0].result, emotion_image_path):
            pipeline_logger.log_file_operation("Saving Emotion Sheet", str(emotion_image_path), "success")
            pipeline_logger.log_sheet_generation("Emotion", "success", str(emotion_image_path))
        else:
            pipeline_logger.log_file_operation("Saving Emotion Sheet", str(emotion_image_path), "error")
            return None
        
        pipeline_logger.log_info(" All character sheets generated successfully!")
        return uncut_dir
        
    except Exception as e:
        pipeline_logger.log_error(f"Error during character sheet generation: {str(e)}")
        return None
