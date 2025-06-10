"""
CLIP Similarity Evaluator

This module uses OpenAI's CLIP model to calculate semantic similarity between images and text descriptions.
It provides functionality to evaluate how well generated images align with their corresponding prompts.
"""

import os
import torch
import open_clip
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPSimilarityEvaluator:
    """
    A class to evaluate semantic similarity between images and text descriptions using CLIP.
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        """
        Initialize the CLIP similarity evaluator.
        
        Args:
            model_name (str): CLIP model variant to use. Options: 'ViT-B-32', 'ViT-B-16', 'ViT-L-14'
            pretrained (str): Pretrained weights to use. 'openai' for OpenAI weights.
            device (str, optional): Device to run the model on. If None, automatically selects GPU if available.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        
        logger.info(f"Loading CLIP model '{model_name}' with '{pretrained}' weights on device '{self.device}'...")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            # Get tokenizer using the model's tokenizer instead of model_name
            self.tokenizer = open_clip.get_tokenizer(model_name) if hasattr(open_clip, 'get_tokenizer') else open_clip.tokenize
            self.model.eval()
            logger.info("CLIP model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            # Try fallback with different model naming
            try:
                logger.info("Trying fallback model configuration...")
                # Use a known working model configuration
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
                )
                self.tokenizer = open_clip.tokenize
                self.model.eval()
                logger.info("CLIP model loaded successfully with fallback configuration!")
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                raise
    
    def encode_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Encode an image into CLIP embeddings.
        
        Args:
            image: Image path (str/Path) or PIL Image object
            
        Returns:
            torch.Tensor: Normalized image embeddings
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image object")
        
        # Preprocess and encode
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text(s) into CLIP embeddings.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            torch.Tensor: Normalized text embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize and encode
        if callable(self.tokenizer):
            # If tokenizer is a function (like open_clip.tokenize)
            text_inputs = self.tokenizer(texts).to(self.device)
        else:
            # If tokenizer is an object with a __call__ method
            text_inputs = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def calculate_similarity(self, image: Union[str, Path, Image.Image], 
                           text: str) -> float:
        """
        Calculate cosine similarity between an image and text description.
        
        Args:
            image: Image path or PIL Image object
            text: Text description
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            
            # Calculate cosine similarity
            similarity = torch.cosine_similarity(image_features, text_features, dim=1)
            return float(similarity.cpu().item())
        
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def batch_evaluate(self, image_text_pairs: List[Tuple[Union[str, Path, Image.Image], str]]) -> List[float]:
        """
        Evaluate similarity for multiple image-text pairs.
        
        Args:
            image_text_pairs: List of (image, text) tuples
            
        Returns:
            List[float]: List of similarity scores
        """
        similarities = []
        
        logger.info(f"Evaluating {len(image_text_pairs)} image-text pairs...")
        
        for i, (image, text) in enumerate(image_text_pairs):
            try:
                similarity = self.calculate_similarity(image, text)
                similarities.append(similarity)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_text_pairs)} pairs")
                    
            except Exception as e:
                logger.error(f"Error processing pair {i}: {str(e)}")
                similarities.append(0.0)
        
        return similarities
    
    def evaluate_directory(self, image_dir: Union[str, Path], 
                          descriptions: Dict[str, str],
                          output_file: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate all images in a directory against their descriptions.
        
        Args:
            image_dir: Directory containing images
            descriptions: Dictionary mapping image filenames to descriptions
            output_file: Optional file to save results as JSON
            
        Returns:
            Dict[str, float]: Dictionary mapping image filenames to similarity scores
        """
        image_dir = Path(image_dir)
        results = {}
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions and f.name in descriptions]
        
        logger.info(f"Found {len(image_files)} images with descriptions in {image_dir}")
        
        for image_file in image_files:
            try:
                description = descriptions[image_file.name]
                similarity = self.calculate_similarity(image_file, description)
                results[image_file.name] = similarity
                
                logger.info(f"{image_file.name}: {similarity:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {str(e)}")
                results[image_file.name] = 0.0
        
        # Calculate statistics
        if results:
            scores = list(results.values())
            avg_similarity = np.mean(scores)
            std_similarity = np.std(scores)
            min_similarity = np.min(scores)
            max_similarity = np.max(scores)
            
            stats = {
                'average_similarity': float(avg_similarity),
                'std_similarity': float(std_similarity),
                'min_similarity': float(min_similarity),
                'max_similarity': float(max_similarity),
                'total_images': len(results)
            }
            
            logger.info(f"\n=== Similarity Statistics ===")
            logger.info(f"Average similarity: {avg_similarity:.3f} ± {std_similarity:.3f}")
            logger.info(f"Range: {min_similarity:.3f} - {max_similarity:.3f}")
            logger.info(f"Total images evaluated: {len(results)}")
            
            # Save results if output file specified
            if output_file:
                output_data = {
                    'evaluation_date': datetime.now().isoformat(),
                    'model_used': self.model_name,
                    'statistics': stats,
                    'individual_scores': results
                }
                
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Results saved to {output_file}")
        
        return results
    
    def evaluate_character_generations(self, character_name: str, 
                                     base_dir: str = "generated_images") -> Dict[str, float]:
        """
        Evaluate generated images for a specific character.
        This method assumes a structure where character descriptions are stored
        in character metadata files.
        
        Args:
            character_name: Name of the character to evaluate
            base_dir: Base directory containing generated images
            
        Returns:
            Dict[str, float]: Dictionary mapping image filenames to similarity scores
        """
        character_dir = Path(base_dir) / character_name
        
        if not character_dir.exists():
            logger.error(f"Character directory not found: {character_dir}")
            return {}
        
        # Try to find character description
        character_info_path = Path("Characters") / character_name / "character_info.json"
        if character_info_path.exists():
            try:
                with open(character_info_path, 'r') as f:
                    character_info = json.load(f)
                    character_description = character_info.get('description', f"an image of {character_name}")
            except Exception as e:
                logger.warning(f"Could not load character info: {str(e)}")
                character_description = f"an image of {character_name}"
        else:
            character_description = f"an image of {character_name}"
        
        # Create descriptions dictionary for all images
        image_files = list(character_dir.glob("*.png")) + list(character_dir.glob("*.jpg"))
        descriptions = {img.name: character_description for img in image_files}
        
        # Evaluate
        output_file = character_dir / f"clip_similarity_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return self.evaluate_directory(character_dir, descriptions, str(output_file))
    
    def evaluate_character_cut_images(self, character_name: str) -> Dict[str, float]:
        """
        Evaluate cut character sheet images for a specific character.
        These are typically character sheets, action sheets, and emotion sheets.
        
        Args:
            character_name: Name of the character to evaluate
            
        Returns:
            Dict[str, float]: Dictionary mapping image filenames to similarity scores
        """
        character_cut_dir = Path("Characters") / character_name / "cut_images"
        
        if not character_cut_dir.exists():
            logger.error(f"Character cut images directory not found: {character_cut_dir}")
            return {}
        
        # Try to find character description
        character_info_path = Path("Characters") / character_name / "model_info.json"
        base_description = f"an image of {character_name}"
        
        if character_info_path.exists():
            try:
                with open(character_info_path, 'r') as f:
                    character_info = json.load(f)
                    # Look for description in various possible keys
                    if 'description' in character_info:
                        base_description = character_info['description']
                    elif 'character_description' in character_info:
                        base_description = character_info['character_description']
                    elif 'prompt' in character_info:
                        base_description = character_info['prompt']
                    else:
                        base_description = f"an image of {character_name}"
            except Exception as e:
                logger.warning(f"Could not load character info: {str(e)}")
                base_description = f"an image of {character_name}"
        
        # Find all image files
        image_files = list(character_cut_dir.glob("*.png")) + list(character_cut_dir.glob("*.jpg"))
        descriptions = {}
        
        # Create specific descriptions based on file types
        for img_file in image_files:
            filename = img_file.name.lower()
            
            if filename.startswith('char_sheet'):
                descriptions[img_file.name] = f"{base_description} character sheet with multiple poses and expressions"
            elif filename.startswith('action_sheet'):
                descriptions[img_file.name] = f"{base_description} action sheet showing different dynamic poses and movements"
            elif filename.startswith('emotion_sheet'):
                descriptions[img_file.name] = f"{base_description} emotion sheet displaying various facial expressions and emotions"
            else:
                # Generic description for any other images
                descriptions[img_file.name] = base_description
        
        logger.info(f"Found {len(descriptions)} cut images for {character_name}")
        
        # Evaluate
        output_file = character_cut_dir / f"clip_similarity_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return self.evaluate_directory(character_cut_dir, descriptions, str(output_file))


def main():
    """
    Example usage of the CLIP Similarity Evaluator.
    """
    # Initialize evaluator
    evaluator = CLIPSimilarityEvaluator(model_name="ViT-B-32")
    
    # Example 1: Single image-text pair
    print("=== Single Image Evaluation ===")
    # Replace with actual image path and description
    sample_image = "path/to/your/image.png"
    sample_description = "a photo of a cartoon character with blue hair"
    
    if os.path.exists(sample_image):
        similarity = evaluator.calculate_similarity(sample_image, sample_description)
        print(f"Similarity score: {similarity:.3f}")
    else:
        print("Sample image not found, skipping single evaluation")
    
    # Example 2: Evaluate a character's generated images
    print("\n=== Character Evaluation ===")
    # Check if we have any character directories
    characters_dir = Path("Characters")
    if characters_dir.exists():
        character_folders = [d.name for d in characters_dir.iterdir() if d.is_dir()]
        if character_folders:
            # Evaluate first character found
            character_name = character_folders[0]
            print(f"Evaluating character: {character_name}")
            results = evaluator.evaluate_character_generations(character_name)
            
            if results:
                avg_score = np.mean(list(results.values()))
                print(f"Average similarity score for {character_name}: {avg_score:.3f}")
            else:
                print(f"No images found for {character_name}")
        else:
            print("No character folders found")
    else:
        print("Characters directory not found")
    
    # Example 3: Custom evaluation
    print("\n=== Custom Evaluation Example ===")
    print("To use this evaluator in your own code:")
    print("""
    from src.clip_similarity_evaluator import CLIPSimilarityEvaluator
    
    # Initialize
    evaluator = CLIPSimilarityEvaluator()
    
    # Single evaluation
    score = evaluator.calculate_similarity('image.png', 'description')
    
    # Batch evaluation
    pairs = [('image1.png', 'desc1'), ('image2.png', 'desc2')]
    scores = evaluator.batch_evaluate(pairs)
    
    # Directory evaluation
    descriptions = {'image1.png': 'desc1', 'image2.png': 'desc2'}
    results = evaluator.evaluate_directory('image_folder/', descriptions, 'results.json')
    """)


if __name__ == "__main__":
    main()
