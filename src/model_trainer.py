"""
This module handles the training of SDXL models using the Replicate API.
"""

import os
import replicate
from pathlib import Path
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from s3_uploader import upload_file_to_s3

# Load environment variables from .env file
load_dotenv()

def train_sdxl_model(character_name, training_images_dir, character_description):
    """
    Train a SDXL model using the provided character images and description.
    
    Args:
        character_name (str): Name of the character
        training_images_dir (Path): Directory containing the training images
        character_description (str): Description of the character
        
    Returns:
        str: The trained model ID if successful, None otherwise
    """
    print("\n=== Training SDXL Model ===")
    
    # Check for Replicate API token
    replicate_token = os.getenv('REPLICATE_API_TOKEN')
    if not replicate_token:
        print("Error: REPLICATE_API_TOKEN not found in .env file")
        print("Please create a .env file in the project root with your Replicate API token:")
        print("REPLICATE_API_TOKEN=your_token_here")
        return None
    
    # Set the API token
    os.environ["REPLICATE_API_TOKEN"] = replicate_token
    
    # Initialize the Replicate client
    replicate_client = replicate.Client(api_token=replicate_token)
    
    # Prepare the training data
    print("Preparing training data...")
    
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_filename = f"{character_name}-{current_datetime}.zip"
    zip_path = Path(f"characters/{character_name}/{zip_filename}")
    # Create a zip file of the training images
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img_path in training_images_dir.glob("*.png"):
            zipf.write(img_path, img_path.name)
    print(f"Training data prepared and saved to {zip_path}")
    
    # Upload zip to S3 and get public link
    print("Uploading zip file to AWS S3...")
    # Set your S3 bucket name here
    s3_bucket = os.getenv('S3_BUCKET_NAME')
    s3_key = zip_filename
    public_url = upload_file_to_s3(str(zip_path), s3_bucket, s3_key)
    if not public_url:
        print("Error: Failed to upload zip file to S3.")
        return None
    print(f"S3 public URL: {public_url}")
    
    try:
        # Create destination model name with character name and date
        model_name = f"{character_name.lower()}-{current_datetime}"
        destination_model = f"nisargbrahmbhatt4778/{model_name}"
        
        # Create a new model
        print(f"Creating new model: {model_name}")
        model = replicate_client.models.create(
            owner="nisargbrahmbhatt4778",
            name=model_name,
            visibility="public",
            hardware="gpu-l40s"
        )
        
        print(f"Model created successfully with ID: {model.id}")
        
        # Start the training
        print("Starting model training...")
        training = replicate.trainings.create(
            destination=destination_model,
            version="stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input={
                "input_images": public_url,
                "token_string": "TOK",
                "caption_prefix": f"a photo of TOK, ",
                "max_train_steps": 1000,
                "use_face_detection_instead": False
            }
        )
        
        # Wait for training to complete
        print("Training started. This may take a while...")
        while training.status == "starting" or training.status == "processing":
            print(f"Training status: {training.status}")
            time.sleep(60)  # Check status every minute
            training.reload()
        
        if training.status == "succeeded":
            print("Training completed successfully!")
            print(f"Training output: {training.output}")
            
            # Extract the version string from training.output
            if isinstance(training.output, dict) and 'version' in training.output:
                model_version = training.output['version']
            else:
                # Fallback: if training.output is already a string, use it directly
                model_version = str(training.output)
            
            print(f"Model version: {model_version}")
            
            # Save the model information
            model_info = {
                "character_name": character_name,
                "model_id": {
                    "version": model_version,
                    "weights": training.output.get('weights') if isinstance(training.output, dict) else None
                },
                "training_description": character_description,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "destination_model": destination_model,
                "replicate_model_id": model.id
            }
            
            info_path = Path(f"characters/{character_name}/model_info.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return model_version
        else:
            print(f"Training failed with status: {training.status}")
            return None
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None
    finally:
        # Clean up the zip file
        if zip_path.exists():
            zip_path.unlink() 