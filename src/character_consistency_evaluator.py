"""
Character Consistency Evaluator

This module uses CLIP to evaluate character consistency between a reference image 
(cut image/character sheet) and generated images where the character may be 
performing different actions or in different contexts.

The goal is to measure how visually similar the character appears across different 
scenes, regardless of pose, action, or objects they're interacting with.
"""

import os
import torch
import clip
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


class CharacterConsistencyEvaluator:
    """
    Evaluates character consistency between reference images and generated images
    using CLIP similarity, focusing on character appearance rather than actions/context.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize the Character Consistency Evaluator.
        
        Args:
            model_name (str): CLIP model variant to use
            device (str, optional): Device to run the model on
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logger.info(f"Loading CLIP model '{model_name}' on device '{self.device}'...")
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def calculate_character_consistency(self, reference_image: Union[str, Path, Image.Image], 
                                      generated_image: Union[str, Path, Image.Image]) -> float:
        """
        Calculate consistency between a reference character image and a generated image.
        
        This measures how similar the character appears, regardless of different actions,
        poses, or objects they might be interacting with.
        
        Args:
            reference_image: Reference/cut image of the character (baseline)
            generated_image: Generated image with the character in different context
            
        Returns:
            float: Consistency score between 0 and 1 (higher = more consistent)
        """
        try:
            # Load and preprocess images
            if isinstance(reference_image, (str, Path)):
                ref_img = Image.open(reference_image).convert('RGB')
            else:
                ref_img = reference_image
                
            if isinstance(generated_image, (str, Path)):
                gen_img = Image.open(generated_image).convert('RGB')
            else:
                gen_img = generated_image
            
            # Preprocess images
            ref_input = self.preprocess(ref_img).unsqueeze(0).to(self.device)
            gen_input = self.preprocess(gen_img).unsqueeze(0).to(self.device)
            
            # Encode images
            with torch.no_grad():
                ref_features = self.model.encode_image(ref_input)
                gen_features = self.model.encode_image(gen_input)
                
                # Normalize features
                ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
                gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(ref_features, gen_features, dim=1)
                
            return float(similarity.cpu().item())
            
        except Exception as e:
            logger.error(f"Error calculating consistency: {str(e)}")
            return 0.0
    
    def evaluate_character_set(self, reference_image: Union[str, Path], 
                              generated_images: List[Union[str, Path]], 
                              character_name: str = "Character") -> Dict:
        """
        Evaluate consistency for a set of generated images against a reference.
        
        Args:
            reference_image: Path to reference/cut image
            generated_images: List of paths to generated images
            character_name: Name of the character for identification
            
        Returns:
            Dict: Comprehensive consistency analysis
        """
        logger.info(f"Evaluating consistency for {character_name} with {len(generated_images)} images...")
        
        results = {
            "character_name": character_name,
            "reference_image": str(reference_image),
            "evaluation_date": datetime.now().isoformat(),
            "model_used": self.model_name,
            "individual_scores": {},
            "statistics": {}
        }
        
        consistency_scores = []
        
        for img_path in generated_images:
            try:
                score = self.calculate_character_consistency(reference_image, img_path)
                results["individual_scores"][str(img_path)] = score
                consistency_scores.append(score)
                
                # Log progress
                status = "Excellent" if score >= 0.25 else "Good" if score >= 0.15 else "Needs Work"
                logger.info(f"{status} - {Path(img_path).name}: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                results["individual_scores"][str(img_path)] = 0.0
                consistency_scores.append(0.0)
        
        # Calculate statistics
        if consistency_scores:
            results["statistics"] = {
                "mean_consistency": float(np.mean(consistency_scores)),
                "std_consistency": float(np.std(consistency_scores)),
                "min_consistency": float(np.min(consistency_scores)),
                "max_consistency": float(np.max(consistency_scores)),
                "median_consistency": float(np.median(consistency_scores)),
                "total_images": len(consistency_scores),
                "excellent_count": sum(1 for s in consistency_scores if s >= 0.25),
                "good_count": sum(1 for s in consistency_scores if 0.15 <= s < 0.25),
                "needs_work_count": sum(1 for s in consistency_scores if s < 0.15)
            }
            
            # Log summary
            stats = results["statistics"]
            logger.info(f"\n=== {character_name} Consistency Summary ===")
            logger.info(f"Average consistency: {stats['mean_consistency']:.4f} ± {stats['std_consistency']:.4f}")
            logger.info(f"Range: {stats['min_consistency']:.4f} - {stats['max_consistency']:.4f}")
            logger.info(f"Quality distribution:")
            logger.info(f"  Excellent (>=0.25): {stats['excellent_count']}/{stats['total_images']} ({stats['excellent_count']/stats['total_images']*100:.1f}%)")
            logger.info(f"  Good (0.15-0.24): {stats['good_count']}/{stats['total_images']} ({stats['good_count']/stats['total_images']*100:.1f}%)")
            logger.info(f"  Needs Work (<0.15): {stats['needs_work_count']}/{stats['total_images']} ({stats['needs_work_count']/stats['total_images']*100:.1f}%)")
        
        return results
    
    def compare_multiple_characters(self, character_configs: List[Dict]) -> Dict:
        """
        Compare consistency across multiple characters.
        
        Args:
            character_configs: List of dicts with 'name', 'reference_image', 'generated_images'
            
        Returns:
            Dict: Comprehensive multi-character analysis
        """
        logger.info(f"Evaluating consistency for {len(character_configs)} characters...")
        
        all_results = {
            "evaluation_date": datetime.now().isoformat(),
            "model_used": self.model_name,
            "character_results": {},
            "comparative_analysis": {}
        }
        
        character_summaries = []
        
        for config in character_configs:
            char_name = config["name"]
            ref_image = config["reference_image"]
            gen_images = config["generated_images"]
            
            # Evaluate this character
            char_results = self.evaluate_character_set(ref_image, gen_images, char_name)
            all_results["character_results"][char_name] = char_results
            
            # Store summary for comparison
            if char_results["statistics"]:
                character_summaries.append({
                    "name": char_name,
                    "mean_consistency": char_results["statistics"]["mean_consistency"],
                    "total_images": char_results["statistics"]["total_images"],
                    "excellent_rate": char_results["statistics"]["excellent_count"] / char_results["statistics"]["total_images"]
                })
        
        # Comparative analysis
        if character_summaries:
            # Sort by consistency
            character_summaries.sort(key=lambda x: x["mean_consistency"], reverse=True)
            
            all_results["comparative_analysis"] = {
                "character_rankings": [(c["name"], c["mean_consistency"]) for c in character_summaries],
                "overall_statistics": {
                    "best_character": character_summaries[0]["name"],
                    "best_consistency": character_summaries[0]["mean_consistency"],
                    "worst_character": character_summaries[-1]["name"],
                    "worst_consistency": character_summaries[-1]["mean_consistency"],
                    "average_consistency": np.mean([c["mean_consistency"] for c in character_summaries]),
                    "total_images_evaluated": sum(c["total_images"] for c in character_summaries)
                }
            }
            
            logger.info(f"\n=== Multi-Character Comparison ===")
            logger.info("Character rankings by consistency:")
            for i, (name, score) in enumerate(all_results["comparative_analysis"]["character_rankings"], 1):
                logger.info(f"{i}. {name}: {score:.4f}")
        
        return all_results
    
    def batch_evaluate_directory(self, characters_dir: Union[str, Path], 
                                generated_images_dir: Union[str, Path], 
                                output_file: Optional[str] = None) -> Dict:
        """
        Batch evaluate character consistency for your directory structure.
        
        Args:
            characters_dir: Directory containing character folders with cut_images
            generated_images_dir: Directory containing generated images organized by character
            output_file: Optional file to save results
            
        Returns:
            Dict: Complete evaluation results
        """
        characters_dir = Path(characters_dir)
        generated_images_dir = Path(generated_images_dir)
        
        logger.info(f"Batch evaluating characters from: {characters_dir}")
        logger.info(f"Generated images from: {generated_images_dir}")
        
        character_configs = []
        
        # Find character directories in Characters folder
        for char_dir in characters_dir.iterdir():
            if not char_dir.is_dir():
                continue
                
            char_name = char_dir.name
            cut_images_dir = char_dir / "cut_images"
            
            # Find reference image (first character sheet)
            reference_image = None
            if cut_images_dir.exists():
                # Look for character sheets
                char_sheets = (list(cut_images_dir.glob("char_sheet_*.png")) + 
                              list(cut_images_dir.glob("char_sheet_*.jpg")) +
                              list(cut_images_dir.glob("character_sheet_*.png")) +
                              list(cut_images_dir.glob("character_sheet_*.jpg")))
                
                if char_sheets:
                    reference_image = char_sheets[0]  # Use first character sheet as reference
                else:
                    # If no char_sheet found, use any image in cut_images as reference
                    all_images = (list(cut_images_dir.glob("*.png")) + 
                                 list(cut_images_dir.glob("*.jpg")) +
                                 list(cut_images_dir.glob("*.jpeg")))
                    if all_images:
                        reference_image = all_images[0]
                        logger.info(f"Using {reference_image.name} as reference for {char_name}")
            
            # Find generated images in generated_images folder
            generated_char_dir = generated_images_dir / char_name
            generated_images = []
            
            if generated_char_dir.exists():
                # Find all image files in the character's generated folder
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    generated_images.extend(generated_char_dir.glob(f"*{ext}"))
            
            if reference_image and generated_images:
                character_configs.append({
                    "name": char_name,
                    "reference_image": str(reference_image),
                    "generated_images": [str(img) for img in generated_images]
                })
                logger.info(f"Found {char_name}: 1 reference + {len(generated_images)} generated images")
            else:
                missing_parts = []
                if not reference_image:
                    missing_parts.append("reference image")
                if not generated_images:
                    missing_parts.append("generated images")
                logger.warning(f"Skipping {char_name}: missing {' and '.join(missing_parts)}")
        
        if not character_configs:
            logger.error("No valid character configurations found!")
            logger.error("Please check your directory structure:")
            logger.error(f"Characters dir: {characters_dir}")
            logger.error(f"Generated images dir: {generated_images_dir}")
            return {}
        
        logger.info(f"Starting evaluation for {len(character_configs)} characters...")
        
        # Evaluate all characters
        results = self.compare_multiple_characters(character_configs)
        
        # Save results if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save results: {str(e)}")
        
        return results


def main():
    """
    Example usage of the Character Consistency Evaluator.
    """
    print("=== Character Consistency Evaluator ===")
    
    # Initialize evaluator
    try:
        evaluator = CharacterConsistencyEvaluator()
    except Exception as e:
        print(f"Failed to initialize evaluator: {str(e)}")
        return
    
    # Example 1: Single character consistency evaluation
    print("\n=== Example: Single Character Evaluation ===")
    
    # Replace these paths with your actual file paths
    reference_image = "Characters/Milo/cut_images/char_sheet_1.png"
    generated_images = [
        "generated_images/Milo/milo_running.png",
        "generated_images/Milo/milo_eating.png", 
        "generated_images/Milo/milo_reading.png"
    ]
    
    # Check if example files exist
    if os.path.exists(reference_image):
        results = evaluator.evaluate_character_set(
            reference_image=reference_image,
            generated_images=generated_images,
            character_name="Milo"
        )
        
        print(f"Mean consistency: {results['statistics']['mean_consistency']:.4f}")
        print(f"Images evaluated: {results['statistics']['total_images']}")
    else:
        print("Example files not found. Please update paths in the code.")
    
    # Example 2: Batch directory evaluation for your structure
    print("\n=== Example: Batch Directory Evaluation ===")
    
    characters_directory = "Characters/"
    generated_images_directory = "generated_images/"
    
    if os.path.exists(characters_directory) and os.path.exists(generated_images_directory):
        results = evaluator.batch_evaluate_directory(
            characters_dir=characters_directory,
            generated_images_dir=generated_images_directory,
            output_file="character_consistency_results.json"
        )
        
        if results and "comparative_analysis" in results:
            comp_analysis = results["comparative_analysis"]
            print(f"Best character: {comp_analysis['overall_statistics']['best_character']}")
            print(f"Best consistency: {comp_analysis['overall_statistics']['best_consistency']:.4f}")
    else:
        print("Directories not found. Please check:")
        print(f"Characters directory: {characters_directory}")
        print(f"Generated images directory: {generated_images_directory}")
    
    # Example 3: Quick single comparison
    print("\n=== Example: Quick Single Comparison ===")
    
    ref_img = "Characters/Milo/cut_images/char_sheet_1.png"
    gen_img = "generated_images/Milo/some_scene.png"
    
    if os.path.exists(ref_img) and os.path.exists(gen_img):
        score = evaluator.calculate_character_consistency(ref_img, gen_img)
        print(f"Consistency score: {score:.4f}")
        
        if score >= 0.25:
            print("Excellent consistency!")
        elif score >= 0.15:
            print("Good consistency")
        else:
            print("Needs improvement")



if __name__ == "__main__":
    main()