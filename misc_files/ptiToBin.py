# ptiToBin.py (or convert_embedding.py)
# THIS VERSION TRIES TO LOAD AS SAFETENSORS FIRST

import torch
import os
from safetensors.torch import load_file as load_safetensors # Import for safetensors

# === Configuration (match this with your main script's setup) ===
script_dir = os.path.dirname(os.path.abspath(__file__))
original_embedding_filename = "embeddings.pti"
# Ensure this path is correct for your structure
original_embedding_folder = os.path.join(script_dir, "Characters/Milo/trained_models/milo-20250522-011112")

original_embedding_path = os.path.join(original_embedding_folder, original_embedding_filename)

converted_embedding_filename = "embeddings_converted.pti" # Output file
converted_embedding_path = os.path.join(original_embedding_folder, converted_embedding_filename)

# This MUST be the trigger token associated with your embeddings
trigger_token = "<s0><s1>"
# ==============================================================

def convert_replicate_embedding():
    if not os.path.isfile(original_embedding_path):
        print(f"Error: Original embedding file not found at {original_embedding_path}")
        return

    original_state_dict = None
    print(f"Attempting to load original embedding from: {original_embedding_path}")

    # Attempt 1: Try loading as a safetensors file
    try:
        print("Trying to load as a safetensors file...")
        original_state_dict = load_safetensors(original_embedding_path, device="cpu")
        print("Successfully loaded with safetensors.torch.load_file!")
    except Exception as e_sf:
        print(f"Failed to load with safetensors: {e_sf}")
        print("Now trying to load with torch.load (standard PyTorch method)...")
        # Attempt 2: Try loading with torch.load (original method)
        try:
            original_state_dict = torch.load(original_embedding_path, map_location="cpu")
            print("Successfully loaded with torch.load!")
        except Exception as e_torch:
            print(f"Failed to load with torch.load: {e_torch}")
            print("The file might be corrupted, not a valid PyTorch or safetensors file, or in an unexpected format.")
            return # Exit if both loading methods fail

    if original_state_dict is None:
        print("Could not load the embedding file with any known method.")
        return

    # Proceed with conversion if loading was successful
    # Ensure the loaded dict has the expected keys from the safetensors file structure
    if 'text_encoders_0' in original_state_dict and 'text_encoders_1' in original_state_dict:
        embedding_te1 = original_state_dict['text_encoders_0']
        embedding_te2 = original_state_dict['text_encoders_1']
        
        new_state_dict = {
            trigger_token: [embedding_te1, embedding_te2]
        }

        print(f"Saving converted embedding to: {converted_embedding_path}")
        torch.save(new_state_dict, converted_embedding_path)
        print("Conversion successful!")
        print(f"You can now update your main script's 'embedding_path' to use '{converted_embedding_filename}'.")
    else:
        print("Error: The loaded embedding file (after successful load) does not contain the expected keys 'text_encoders_0' and 'text_encoders_1'.")
        print(f"Found keys in loaded dictionary: {list(original_state_dict.keys())}")

if __name__ == "__main__":
    # Before running, ensure safetensors is installed:
    # pip install safetensors
    # Or for your venv:
    # /Users/nisarg/Desktop/Automated-Story-Illustration/venv/bin/pip install safetensors
    convert_replicate_embedding()