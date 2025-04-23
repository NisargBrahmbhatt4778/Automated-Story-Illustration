from llama_cpp import Llama
import json
import torch
from diffusers import DiffusionPipeline
from img_gen import generate_sdxl_image
from db_manager import CharacterDB
import os
from datetime import datetime

# Initialize database
db = CharacterDB()

# Path to the LLaMA 2 13B GGUF model
MODEL_PATH = "models/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf"
DIFF_MODEL_PATH = "models/sdxl-base-1.0"

# Load LLaMA 2 13B
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=10)

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

def extract_character_features(description):
    """
    Uses LLaMA 2 13B running locally to extract structured character attributes.
    Ensures deterministic JSON output.
    """
    prompt = f"""
    You are a highly accurate AI that extracts structured attributes from character descriptions.
    Your task is to analyze the character description and fill in the JSON template below.
    You must ONLY output the JSON object, nothing else.
    Do not include any explanations, headers, or additional text.
    The output must start with {{ and end with }}.
    
    Here is the JSON template to fill in:

    {{
        "body": {{
            "shape": "",
            "height": "",
            "limbs": "",
            "head_ratio": ""
        }},
        "facial_features": {{
            "eyes": "",
            "eyebrows": "",
            "nose": "",
            "mouth": "",
            "ears": ""
        }},
        "hair": {{
            "style": "",
            "color": "",
            "head_accessories": ""
        }},
        "clothing": {{
            "upper_wear": "",
            "lower_wear": "",
            "footwear": "",
            "handwear": "",
            "accessories": [""]
        }},
        "special_features": {{
            "skin_texture": "",
            "skin_color": "",
            "tattoos_markings": "",
            "extra_limbs": "",
            "wings": "",
            "tail": ""
        }},
        "pose_personality": {{
            "pose": "",
            "expression": "",
            "personality": ""
        }}
    }}

    Now, analyze this character description and fill in the JSON template:
    {description}

    Remember: Output ONLY the completed JSON object, starting with {{ and ending with }}.
    """

    print(f"Prompt sent to LLaMA:\n{prompt}")

    try:
        # Run LLaMA inference with lower temperature for more deterministic output
        response = llm(prompt, max_tokens=1024, stop=["\n\n"], temperature=0.1)
        response_text = response["choices"][0]["text"]
        print(f"Raw LLM Response: {response_text}")

        # Extract JSON from response
        try:
            # Find the first { and last } to extract JSON
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON object found in response")
                
            json_str = response_text[start_idx:end_idx + 1]
            print(f"Extracted JSON string: {json_str}")
            
            # Try to parse the JSON
            structured_data = json.loads(json_str)
            
            # Validate the structure
            required_fields = ["body", "facial_features", "hair", "clothing", "special_features", "pose_personality"]
            for field in required_fields:
                if field not in structured_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return structured_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Problematic JSON string: {json_str}")
            return {"error": f"JSON parsing error: {str(e)}"}
        except ValueError as e:
            print(f"Value error: {str(e)}")
            return {"error": str(e)}
            
    except Exception as e:
        print(f"Error during LLM inference: {str(e)}")
        return {"error": f"LLM inference error: {str(e)}"}

def save_character_json(description, filename="character.json"):
    """
    Generates and saves a deterministic JSON file from a character description.
    """
    try:
        character_data = extract_character_features(description)
        
        if "error" in character_data:
            print(f"Error in character data: {character_data['error']}")
            return False
            
        with open(filename, "w") as file:
            json.dump(character_data, file, indent=4)
            
        print(f"Character JSON successfully saved as {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
        return False


def generate_stable_diffusion_prompt(character_json):
    """
    Uses LLaMA 2 13B to generate a high-quality Stable Diffusion prompt from character JSON.
    """

    prompt = f"""
    You are an AI prompt generator for Stable Diffusion.
    Your task is to convert a structured character description (JSON format) into a **very simple, child-friendly illustration prompt** suitable for a children's storybook.

    The illustration should be:
    - Extremely simple and clean
    - Use basic shapes and minimal details
    - Have a hand-drawn, crayon-like quality
    - Use bright, primary colors
    - Have thick, bold outlines
    - Be suitable for young children (ages 3-6)
    - Have a friendly, approachable style
    - Use minimal shading and simple textures

    ## Example of an ideal output:
    "A simple, friendly robot with glowing red eyes and flowing silver hair.
    He wears basic black armor and a red cape.
    The illustration is drawn with thick, bold lines and uses bright, primary colors.
    The style is very simple, like a child's drawing, with minimal details and shading.
    The background is plain white to keep focus on the character.
    The overall style is clean, friendly, and perfect for a children's storybook."

    ## Character JSON:
    {json.dumps(character_json, indent=4)}

    ## Now generate a **very simple, child-friendly illustration prompt**:
    """

    # Run LLaMA inference (with temperature=0 for deterministic results)
    response = llm(prompt, max_tokens=1024, stop=["\n\n"], temperature=0.1)
    response_text = response["choices"][0]["text"]

    return response_text.strip()







# Example character description
description_text = """
A Robot warrior with glowing red eyes and long flowing silver hair.
He wears heavy black armor and a crimson cape.
His expression is fierce, and he stands tall, exuding power.
"""

# Extract and save JSON
save_character_json(description_text)

# Read character JSON from file
with open("character.json", "r") as file:
    character_json = json.load(file)

# Generate Stable Diffusion prompt
stable_diffusion_prompt = generate_stable_diffusion_prompt(character_json)
print(stable_diffusion_prompt)

# Generate unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_filename = f"images/character_{timestamp}.png"

# Generate SDXL Image
print("Generating SDXL Image...")
generate_sdxl_image(stable_diffusion_prompt, image_filename)

# Save to database
db.save_character(
    description=description_text,
    json_data=json.dumps(character_json),
    image_path=image_filename
)

print("Character record saved to database")