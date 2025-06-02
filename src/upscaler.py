"""
This module handles the upscaling of character images using Real-ESRGAN.
"""

import os
from pathlib import Path
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
import cv2
import numpy as np
from PIL import Image

def setup_upscaler():
    """
    Set up the Real-ESRGAN upscaler with the Anime6B model.
    
    Returns:
        RealESRGANer: Configured upscaler instance
    """
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    model_path = str(Path('models/RealESRGAN_x4plus_anime_6B.pth'))

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        device=device,
        tile=0,
        pre_pad=0,
        half=False
    )
    
    return upsampler

def upscale_image(image_path, upsampler, output_path):
    """
    Upscale a single image using Real-ESRGAN.
    
    Args:
        image_path (str): Path to the input image
        upsampler (RealESRGANer): Configured upscaler instance
        output_path (str): Path to save the upscaled image
    """
    try:
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return False
            
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Upscale
        output, _ = upsampler.enhance(img, outscale=4)
        
        # Convert back to BGR for saving
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Save the upscaled image
        cv2.imwrite(str(output_path), output)
        return True
        
    except Exception as e:
        print(f"Error upscaling image {image_path}: {str(e)}")
        return False

def upscale_character_images(character_name, input_dir):
    """
    Upscale all images in the input directory and save them to a new upscaled directory
    within the character's directory.
    
    Args:
        character_name (str): Name of the character
        input_dir (Path): Path to the directory containing images to upscale
        
    Returns:
        Path: Path to the directory containing upscaled images, or None if failed
    """
    print("\n=== Upscaling Character Images ===")
    
    # Create output directory within the character's directory
    output_dir = Path(f"characters/{character_name}/upscaled_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the upscaler
    try:
        upsampler = setup_upscaler()
    except Exception as e:
        print(f"Failed to set up upscaler: {str(e)}")
        return None
    
    # Process each image
    success_count = 0
    total_images = 0
    
    for image_path in input_dir.glob("*.png"):
        total_images += 1
        output_path = output_dir / image_path.name
        
        if upscale_image(image_path, upsampler, output_path):
            success_count += 1
            print(f"Successfully upscaled: {image_path.name}")
    
    if success_count == total_images:
        print(f"\nSuccessfully upscaled all {success_count} images!")
        return output_dir
    else:
        print(f"\nUpscaled {success_count} out of {total_images} images.")
        return output_dir if success_count > 0 else None 