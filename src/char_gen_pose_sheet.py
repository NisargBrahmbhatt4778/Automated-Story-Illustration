import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import argparse
from datetime import datetime
import os

class CharacterSheetGenerator:
    def __init__(self):
        # Optimize device selection for M3 Max
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA")
        else:
            self.device = "cpu"
            print("Using CPU")
            
        self.pipeline = None
        
    def load_models(self):
        """Load Flux model optimized for M3 Max"""
        print("Loading Flux pipeline...")
        
        try:
            # Load Flux pipeline with M3 Max optimizations
            self.pipeline = DiffusionPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16,  # Use float16 for MPS compatibility
                variant="fp16",  # Use fp16 variant for better memory usage
                use_safetensors=True,
                device_map=None  # Don't use device_map with MPS
            )
            
            # Move to MPS device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention and other optimizations
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing(1)
                print("Enabled attention slicing")
                
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                # Don't use CPU offload with 64GB RAM - keep everything in memory
                pass
                
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
                print("Enabled VAE slicing")
                
            # Set memory format for M3 Max optimization
            if self.device == "mps":
                torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
                
            print("Flux model loaded successfully on M3 Max!")
            
        except Exception as e:
            print(f"Error loading Flux model: {e}")
            print("Trying SDXL as fallback...")
            
            # Fallback to SDXL with M3 Max optimizations
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                    device_map=None
                )
                
                self.pipeline = self.pipeline.to(self.device)
                
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing(1)
                if hasattr(self.pipeline, 'enable_vae_slicing'):
                    self.pipeline.enable_vae_slicing()
                    
                print("Loaded SDXL as fallback model on M3 Max")
                
            except Exception as e2:
                raise RuntimeError(f"Could not load any model. Flux error: {e}, SDXL error: {e2}")
        
    def preprocess_pose_image(self, pose_image_path):
        """Preprocess the pose reference image"""
        if not os.path.exists(pose_image_path):
            raise FileNotFoundError(f"Pose image not found: {pose_image_path}")
            
        # Load and resize pose image
        pose_image = load_image(pose_image_path)
        pose_image = pose_image.resize((1280, 1280), Image.Resampling.LANCZOS)
        
        # Convert to PIL Image if needed
        if not isinstance(pose_image, Image.Image):
            pose_image = Image.fromarray(pose_image)
        
        return pose_image
        
    def generate_character_sheet(self, 
                                prompt=None, 
                                pose_image_path=None,
                                negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy",
                                num_inference_steps=20,
                                guidance_scale=3.5,
                                seed=None):
        """Generate character sheet image optimized for M3 Max"""
        
        if self.pipeline is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        # Default prompt similar to ComfyUI workflow
        if prompt is None:
            prompt = ("a character sheet, white background, multiple views, from multiple angles, "
                     "visible face, A Robot warrior with glowing red eyes and long flowing silver hair. "
                     "He wears heavy black armor and a crimson cape. His expression is fierce, "
                     "and he stands tall, exuding power.")
        
        # Set seed for reproducibility with MPS support
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "mps":
                # MPS doesn't have separate seed function
                torch.manual_seed(seed)
            elif torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
        print(f"Generating character sheet with prompt: {prompt[:100]}...")
        
        # Prepare generation parameters optimized for M3 Max
        generation_kwargs = {
            "prompt": prompt,
            "height": 1280,
            "width": 1280,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Check pipeline capabilities and add parameters accordingly
        pipeline_signature = self.pipeline.__call__.__code__.co_varnames
        
        if 'negative_prompt' in pipeline_signature:
            generation_kwargs["negative_prompt"] = negative_prompt
            
        if 'max_sequence_length' in pipeline_signature:
            generation_kwargs["max_sequence_length"] = 512
            
        # For Flux models, add joint_attention_kwargs if available
        if hasattr(self.pipeline, 'transformer') and 'joint_attention_kwargs' in pipeline_signature:
            generation_kwargs["joint_attention_kwargs"] = {"scale": 1.0}
        
        # Note: Pose control is temporarily disabled
        if pose_image_path:
            print(f"Note: Pose image provided ({pose_image_path}) but ControlNet not available in this version")
        
        # Generate image with M3 Max optimizations
        try:
            # Clear MPS cache before generation
            if self.device == "mps":
                torch.mps.empty_cache()
                
            with torch.inference_mode():
                # Use autocast for better performance on M3 Max
                if self.device == "mps":
                    with torch.autocast(device_type="cpu", dtype=torch.float16):
                        result = self.pipeline(**generation_kwargs)
                else:
                    result = self.pipeline(**generation_kwargs)
                
            return result.images[0]
            
        except Exception as e:
            print(f"Generation error with full parameters: {e}")
            print("Trying with basic parameters...")
            
            # Fallback with minimal parameters
            try:
                if self.device == "mps":
                    torch.mps.empty_cache()
                    
                basic_kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": min(num_inference_steps, 15),
                    "height": 1024,  # Reduce size for compatibility
                    "width": 1024,
                }
                
                if 'guidance_scale' in pipeline_signature:
                    basic_kwargs["guidance_scale"] = guidance_scale
                
                with torch.inference_mode():
                    result = self.pipeline(**basic_kwargs)
                    
                return result.images[0]
                
            except Exception as e2:
                raise RuntimeError(f"Generation failed with both full and basic parameters. Error: {e2}")
        
    def save_image(self, image, output_dir="outputs", filename=None):
        """Save generated image"""
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"character_sheet_{timestamp}.png"
            
        output_path = os.path.join(output_dir, filename)
        
        # Optimize image saving for large images
        if hasattr(image, 'save'):
            image.save(output_path, "PNG", optimize=True)
        else:
            # Convert if necessary
            Image.fromarray(image).save(output_path, "PNG", optimize=True)
        
        print(f"Character sheet saved to: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate character sheets using Flux model on M3 Max")
    parser.add_argument("--prompt", type=str, help="Text prompt for character generation")
    parser.add_argument("--pose-image", type=str, help="Path to pose reference image")
    parser.add_argument("--negative-prompt", type=str, 
                       default="blurry, low quality, distorted, deformed, ugly, bad anatomy",
                       help="Negative prompt")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--filename", type=str, help="Output filename")
    
    args = parser.parse_args()
    
    # Print system info
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("Running on Apple Silicon with Metal Performance Shaders")
    print("==============================")
    
    # Initialize generator
    print("Initializing Character Sheet Generator for M3 Max...")
    generator = CharacterSheetGenerator()
    
    try:
        # Load models
        generator.load_models()
        
        # Generate character sheet
        image = generator.generate_character_sheet(
            prompt=args.prompt,
            pose_image_path=args.pose_image,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
        
        # Save result
        output_path = generator.save_image(
            image, 
            output_dir=args.output_dir, 
            filename=args.filename
        )
        
        print("Character sheet generation completed successfully on M3 Max!")
        
        # Clean up memory
        if generator.device == "mps":
            torch.mps.empty_cache()
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())