"""
CLIP Similarity Evaluator - Corrected Version

This module uses OpenAI's original CLIP model to calculate semantic similarity between images and text descriptions.
It provides functionality to evaluate how well generated images align with their corresponding prompts.

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


class CLIPSimilarityEvaluator:
    """
    A class to evaluate semantic similarity between images and text descriptions using CLIP.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize the CLIP similarity evaluator.
        
        Args:
            model_name (str): CLIP model variant to use. Defaults to "ViT-B/32".
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logger.info(f"Loading CLIP model '{model_name}' on device '{self.device}'...")
        
        try:
            # Load the original OpenAI CLIP model
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully!")
            
            # Test the model with a simple case to verify it's working
            self._verify_model()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory. Trying to load on CPU...")
                self.device = "cpu"
                self.model, self.preprocess = clip.load(model_name, device="cpu")
                self.model.eval()
                logger.info("CLIP model loaded successfully on CPU!")
            else:
                raise e
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def _verify_model(self):
        """Verify the model is working correctly with a simple test."""
        try:
            # Create a simple test case
            test_image = Image.new('RGB', (224, 224), color='red')
            test_text = "red"
            
            # Test encoding
            image_features = self.encode_image(test_image)
            text_features = self.encode_text(test_text)
            
            # Calculate similarity
            similarity = torch.cosine_similarity(image_features, text_features, dim=1).item()
            
            logger.info(f"Model verification test similarity: {similarity:.4f}")
            
            if similarity < 0.05:
                logger.warning("Model verification gave unexpectedly low similarity. This might indicate an issue.")
            else:
                logger.info("Model verification successful!")
                
        except Exception as e:
            logger.warning(f"Model verification failed: {str(e)}")
    
    def encode_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Encode an image into CLIP embeddings.
        
        Args:
            image: Image path (str/Path) or PIL Image object
            
        Returns:
            torch.Tensor: Normalized image embeddings
        """
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                logger.error(f"Failed to load image {image}: {str(e)}")
                raise
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image object")
        
        # Preprocess and encode
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features
            
        except Exception as e:
            logger.error(f"Failed to encode image: {str(e)}")
            raise
    
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
        
        try:
            # Tokenize text
            text_inputs = clip.tokenize(texts, truncate=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs)
                # Normalize features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features
            
        except Exception as e:
            logger.error(f"Failed to encode text '{texts}': {str(e)}")
            raise
    
    def calculate_similarity(self, image: Union[str, Path, Image.Image], text: str) -> float:
        """
        Calculate cosine similarity between an image and text description.
        
        Args:
            image: Image path or PIL Image object
            text: Text description
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Encode image and text
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            
            # Calculate cosine similarity
            similarity = torch.cosine_similarity(image_features, text_features, dim=1)
            similarity_score = float(similarity.cpu().item())
            
            # Ensure score is in valid range
            similarity_score = max(0.0, min(1.0, similarity_score))
            
            return similarity_score
        
        except Exception as e:
            logger.error(f"Error calculating similarity for text '{text}': {str(e)}")
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
                    logger.info(f"Processed {i + 1}/{len(image_text_pairs)} pairs. Last score: {similarity:.4f}")
                    
            except Exception as e:
                logger.error(f"Error processing pair {i} (text: '{text}'): {str(e)}")
                similarities.append(0.0)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            logger.info(f"Batch evaluation complete. Average similarity: {avg_similarity:.4f}")
        
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
        
        if not image_dir.exists():
            logger.error(f"Image directory not found: {image_dir}")
            return results
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions and f.name in descriptions]
        
        logger.info(f"Found {len(image_files)} images with descriptions in {image_dir}")
        
        if not image_files:
            logger.warning("No matching images found!")
            return results
        
        for image_file in image_files:
            try:
                description = descriptions[image_file.name]
                similarity = self.calculate_similarity(image_file, description)
                results[image_file.name] = similarity
                
                # Provide interpretive feedback
                if similarity >= 0.3:
                    status = "Excellent"
                elif similarity >= 0.2:
                    status = "Good"
                elif similarity >= 0.1:
                    status = "Fair"
                else:
                    status = "Poor"
                
                logger.info(f"{status} - {image_file.name}: {similarity:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {str(e)}")
                results[image_file.name] = 0.0
        
        # Calculate and display statistics
        if results:
            scores = list(results.values())
            avg_similarity = np.mean(scores)
            std_similarity = np.std(scores)
            min_similarity = np.min(scores)
            max_similarity = np.max(scores)
            median_similarity = np.median(scores)
            
            # Count quality tiers
            excellent_count = sum(1 for s in scores if s >= 0.3)
            good_count = sum(1 for s in scores if 0.2 <= s < 0.3)
            fair_count = sum(1 for s in scores if 0.1 <= s < 0.2)
            poor_count = sum(1 for s in scores if s < 0.1)
            
            stats = {
                'average_similarity': float(avg_similarity),
                'median_similarity': float(median_similarity),
                'std_similarity': float(std_similarity),
                'min_similarity': float(min_similarity),
                'max_similarity': float(max_similarity),
                'total_images': len(results),
                'excellent_images': excellent_count,
                'good_images': good_count,
                'fair_images': fair_count,
                'poor_images': poor_count
            }
            
            logger.info(f"\n=== Similarity Statistics ===")
            logger.info(f"Average similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
            logger.info(f"Median similarity: {median_similarity:.4f}")
            logger.info(f"Range: {min_similarity:.4f} - {max_similarity:.4f}")
            logger.info(f"Quality distribution:")
            logger.info(f"  Excellent (≥0.30): {excellent_count}/{len(results)} ({excellent_count/len(results)*100:.1f}%)")
            logger.info(f"  Good (0.20-0.29): {good_count}/{len(results)} ({good_count/len(results)*100:.1f}%)")
            logger.info(f"  Fair (0.10-0.19): {fair_count}/{len(results)} ({fair_count/len(results)*100:.1f}%)")
            logger.info(f"  Poor (<0.10): {poor_count}/{len(results)} ({poor_count/len(results)*100:.1f}%)")
            
            # Save results if output file specified
            if output_file:
                output_data = {
                    'evaluation_date': datetime.now().isoformat(),
                    'model_used': self.model_name,
                    'device_used': self.device,
                    'statistics': stats,
                    'individual_scores': results
                }
                
                try:
                    with open(output_file, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    logger.info(f"Results saved to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save results: {str(e)}")
        
        return results
    
    def evaluate_character_generations(self, character_name: str, 
                                     base_dir: str = "generated_images", 
                                     allow_cut_images: bool = True) -> Dict[str, float]:
        """
        Evaluate generated images for a specific character.
        
        Args:
            character_name: Name of the character to evaluate
            base_dir: Base directory containing generated images
            allow_cut_images: If True, also try to find cut images from training pipeline
            
        Returns:
            Dict[str, float]: Dictionary mapping image filenames to similarity scores
        """
        character_dir = Path(base_dir) / character_name
        
        # If the primary directory doesn't exist and allow_cut_images is True,
        # try the training cut_images directory
        if not character_dir.exists() and allow_cut_images:
            cut_images_dir = Path("Characters") / character_name / "cut_images"
            if cut_images_dir.exists():
                logger.info(f"Using cut images from training pipeline: {cut_images_dir}")
                character_dir = cut_images_dir
            else:
                logger.error(f"Neither generated images nor cut images directory found for character: {character_name}")
                return {}
        elif not character_dir.exists():
            logger.error(f"Character directory not found: {character_dir}")
            return {}
        
        character_description = self._get_character_description(character_name)
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        all_image_files = []
        for ext in image_extensions:
            all_image_files.extend(character_dir.glob(f"*{ext}"))
        
        # Filter out emotion sheet cuts - only include character and action sheet cuts
        image_files = []
        filtered_out_count = 0
        
        for img_file in all_image_files:
            # Skip emotion sheet cuts (emotion_sheet_N.png)
            if img_file.name.startswith('emotion_sheet_'):
                filtered_out_count += 1
                logger.debug(f"Filtering out emotion sheet: {img_file.name}")
                continue
            
            image_files.append(img_file)
        
        if filtered_out_count > 0:
            logger.info(f"Filtered out {filtered_out_count} emotion sheet images from evaluation")
        
        if not image_files:
            logger.warning(f"No character or action sheet images found in {character_dir} (after filtering)")
            if filtered_out_count > 0:
                logger.info(f"Note: {filtered_out_count} emotion sheet images were excluded from evaluation")
            return {}
        
        # Create descriptions dictionary for all remaining images
        descriptions = {img.name: character_description for img in image_files}
        
        logger.info(f"Evaluating {len(image_files)} character and action sheet images for character '{character_name}' (emotion sheets excluded)")
        logger.info(f"Using description: '{character_description}'")
        
        # Evaluate
        output_file = character_dir / f"clip_similarity_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return self.evaluate_directory(character_dir, descriptions, str(output_file))
    
    def _get_character_description(self, character_name: str) -> str:
        """
        Try to find character description from various possible file locations.
        Prioritizes the new character_info.json file created by the main pipeline.
        
        Args:
            character_name: Name of the character
            
        Returns:
            str: Character description or fallback description
        """
        primary_path = Path("Characters") / character_name / "character_info.json"
        
        if primary_path.exists():
            try:
                with open(primary_path, 'r', encoding='utf-8') as f:
                    character_info = json.load(f)
                
                # Check for character_description field
                if 'character_description' in character_info and character_info['character_description']:
                    description = character_info['character_description'].strip()
                    if description:
                        logger.info(f"Found character description in {primary_path}")
                        return description
                
            except Exception as e:
                logger.warning(f"Could not load character info from {primary_path}: {str(e)}")
        
        # Fallback locations for legacy compatibility
        fallback_paths = [
            Path("Characters") / character_name / "model_info.json",
            Path("Characters") / character_name / "info.json",
            Path(character_name) / "character_info.json",
            Path(character_name) / "model_info.json"
        ]
        
        for info_path in fallback_paths:
            if info_path.exists():
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        character_info = json.load(f)
                    
                    # Look for description in various possible keys
                    description_keys = ['training_description', 'description', 'character_description', 'prompt', 'text', 'caption']
                    for key in description_keys:
                        if key in character_info and character_info[key]:
                            description = character_info[key].strip()
                            if description:
                                logger.info(f"Found character description in {info_path} (fallback)")
                                return description
                    
                except Exception as e:
                    logger.warning(f"Could not load character info from {info_path}: {str(e)}")
                    continue
        
        # Fallback description
        fallback_description = f"an illustration of {character_name}, a cartoon character"
        logger.info(f"Using fallback description for {character_name}")
        return fallback_description
    
    def debug_single_evaluation(self, image_path: str, text: str) -> Dict:
        """
        Debug a single image-text evaluation with detailed output.
        
        Args:
            image_path: Path to the image
            text: Text description
            
        Returns:
            Dict: Detailed debug information
        """
        logger.info(f"=== Debug Evaluation ===")
        logger.info(f"Image: {image_path}")
        logger.info(f"Text: '{text}'")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            # Check if image exists
            if not Path(image_path).exists():
                return {"error": f"Image file not found: {image_path}"}
            
            # Load and check image
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Image size: {image.size}")
            logger.info(f"Image mode: {image.mode}")
            
            # Encode image
            image_features = self.encode_image(image)
            logger.info(f"Image features shape: {image_features.shape}")
            logger.info(f"Image features norm: {image_features.norm():.4f}")
            
            # Encode text
            text_features = self.encode_text(text)
            logger.info(f"Text features shape: {text_features.shape}")
            logger.info(f"Text features norm: {text_features.norm():.4f}")
            
            # Calculate similarity
            similarity = torch.cosine_similarity(image_features, text_features, dim=1).item()
            logger.info(f"Cosine similarity: {similarity:.6f}")
            
            # Alternative similarity calculation for comparison
            dot_product = torch.sum(image_features * text_features, dim=1).item()
            logger.info(f"Dot product: {dot_product:.6f}")
            
            # Interpretation
            if similarity >= 0.3:
                interpretation = "Excellent alignment"
            elif similarity >= 0.2:
                interpretation = "Good alignment"
            elif similarity >= 0.1:
                interpretation = "Fair alignment"
            else:
                interpretation = "Poor alignment"
            
            logger.info(f"Interpretation: {interpretation}")
            
            return {
                "similarity": similarity,
                "dot_product": dot_product,
                "interpretation": interpretation,
                "image_features_norm": float(image_features.norm()),
                "text_features_norm": float(text_features.norm()),
                "image_size": image.size,
                "model": self.model_name,
                "device": self.device
            }
            
        except Exception as e:
            error_msg = f"Debug evaluation failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def evaluate_all_characters_cut_images(self, base_characters_dir: str = "Characters") -> Dict[str, Dict[str, float]]:
        """
        Evaluate all images in cut_images directories for all characters.
        
        Args:
            base_characters_dir: Base directory containing all character folders
            
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary with character names as keys,
                                        and inner dictionaries mapping image filenames to similarity scores
        """
        base_dir = Path(base_characters_dir)
        
        if not base_dir.exists():
            logger.error(f"Characters directory not found: {base_dir}")
            return {}
        
        # Find all character directories
        character_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not character_dirs:
            logger.error(f"No character directories found in {base_dir}")
            return {}
        
        logger.info(f"Found {len(character_dirs)} character directories: {[d.name for d in character_dirs]}")
        
        all_results = {}
        overall_scores = []
        total_images_processed = 0
        
        for char_dir in character_dirs:
            character_name = char_dir.name
            logger.info(f"\n=== Evaluating Character: {character_name} ===")
            
            # Look for cut_images directory
            cut_images_dir = char_dir / "cut_images"
            
            if not cut_images_dir.exists():
                logger.warning(f"No cut_images directory found for {character_name}")
                all_results[character_name] = {}
                continue
            
            # Get character description
            character_description = self._get_character_description(character_name)
            
            # Find all image files in cut_images directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
            all_image_files = []
            for ext in image_extensions:
                all_image_files.extend(cut_images_dir.glob(f"*{ext}"))
            
            if not all_image_files:
                logger.warning(f"No images found in {cut_images_dir}")
                all_results[character_name] = {}
                continue
            
            # Filter out debug images and subdirectories
            image_files = []
            for img_file in all_image_files:
                # Skip debug files and images in subdirectories
                if img_file.parent != cut_images_dir:
                    continue
                if img_file.name.startswith('debug_') or img_file.name.startswith('_debug'):
                    continue
                image_files.append(img_file)
            
            if not image_files:
                logger.warning(f"No valid images found in {cut_images_dir} (after filtering)")
                all_results[character_name] = {}
                continue
            
            logger.info(f"Evaluating {len(image_files)} images for {character_name}")
            logger.info(f"Using description: '{character_description}'")
            
            # Create descriptions dictionary for all images
            descriptions = {img.name: character_description for img in image_files}
            
            # Evaluate images for this character
            character_results = self.evaluate_directory(cut_images_dir, descriptions)
            all_results[character_name] = character_results
            
            if character_results:
                scores = list(character_results.values())
                overall_scores.extend(scores)
                total_images_processed += len(scores)
                
                # Character-specific statistics
                avg_score = np.mean(scores)
                excellent_count = sum(1 for s in scores if s >= 0.3)
                good_count = sum(1 for s in scores if 0.2 <= s < 0.3)
                fair_count = sum(1 for s in scores if 0.1 <= s < 0.2)
                poor_count = sum(1 for s in scores if s < 0.1)
                
                logger.info(f"Character {character_name} Summary:")
                logger.info(f"  Average similarity: {avg_score:.4f}")
                logger.info(f"  Total images: {len(scores)}")
                logger.info(f"  Quality distribution:")
                logger.info(f"    Excellent (≥0.30): {excellent_count}/{len(scores)} ({excellent_count/len(scores)*100:.1f}%)")
                logger.info(f"    Good (0.20-0.29): {good_count}/{len(scores)} ({good_count/len(scores)*100:.1f}%)")
                logger.info(f"    Fair (0.10-0.19): {fair_count}/{len(scores)} ({fair_count/len(scores)*100:.1f}%)")
                logger.info(f"    Poor (<0.10): {poor_count}/{len(scores)} ({poor_count/len(scores)*100:.1f}%)")
        
        # Overall statistics
        if overall_scores:
            logger.info(f"\n=== OVERALL STATISTICS FOR ALL CHARACTERS ===")
            overall_avg = np.mean(overall_scores)
            overall_std = np.std(overall_scores)
            overall_median = np.median(overall_scores)
            overall_min = np.min(overall_scores)
            overall_max = np.max(overall_scores)
            
            overall_excellent = sum(1 for s in overall_scores if s >= 0.3)
            overall_good = sum(1 for s in overall_scores if 0.2 <= s < 0.3)
            overall_fair = sum(1 for s in overall_scores if 0.1 <= s < 0.2)
            overall_poor = sum(1 for s in overall_scores if s < 0.1)
            
            logger.info(f"Total characters evaluated: {len([k for k, v in all_results.items() if v])}")
            logger.info(f"Total images evaluated: {total_images_processed}")
            logger.info(f"Overall average similarity: {overall_avg:.4f} ± {overall_std:.4f}")
            logger.info(f"Overall median similarity: {overall_median:.4f}")
            logger.info(f"Overall range: {overall_min:.4f} - {overall_max:.4f}")
            logger.info(f"Overall quality distribution:")
            logger.info(f"  Excellent (≥0.30): {overall_excellent}/{len(overall_scores)} ({overall_excellent/len(overall_scores)*100:.1f}%)")
            logger.info(f"  Good (0.20-0.29): {overall_good}/{len(overall_scores)} ({overall_good/len(overall_scores)*100:.1f}%)")
            logger.info(f"  Fair (0.10-0.19): {overall_fair}/{len(overall_scores)} ({overall_fair/len(overall_scores)*100:.1f}%)")
            logger.info(f"  Poor (<0.10): {overall_poor}/{len(overall_scores)} ({overall_poor/len(overall_scores)*100:.1f}%)")
            
            # Character ranking
            logger.info(f"\n=== CHARACTER RANKINGS ===")
            character_averages = {}
            for char_name, results in all_results.items():
                if results:
                    character_averages[char_name] = np.mean(list(results.values()))
            
            if character_averages:
                sorted_characters = sorted(character_averages.items(), key=lambda x: x[1], reverse=True)
                for i, (char_name, avg_score) in enumerate(sorted_characters, 1):
                    status = "EXCELLENT" if avg_score >= 0.3 else "GOOD" if avg_score >= 0.2 else "POOR"
                    image_count = len(all_results[char_name])
                    logger.info(f"{i:2d}. {status} {char_name:15s}: {avg_score:.4f} ({image_count} images)")
            
            # Save comprehensive results
            output_file = f"all_characters_clip_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_data = {
                'evaluation_date': datetime.now().isoformat(),
                'model_used': self.model_name,
                'device_used': self.device,
                'total_characters': len([k for k, v in all_results.items() if v]),
                'total_images': total_images_processed,
                'overall_statistics': {
                    'average_similarity': float(overall_avg),
                    'median_similarity': float(overall_median),
                    'std_similarity': float(overall_std),
                    'min_similarity': float(overall_min),
                    'max_similarity': float(overall_max),
                    'excellent_images': overall_excellent,
                    'good_images': overall_good,
                    'fair_images': overall_fair,
                    'poor_images': overall_poor
                },
                'character_results': all_results,
                'character_rankings': sorted_characters if 'sorted_characters' in locals() else []
            }
            
            try:
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Comprehensive results saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save comprehensive results: {str(e)}")
        
        return all_results

    # ...existing code...
def main():
    """
    Example usage of the corrected CLIP Similarity Evaluator.
    """
    print("=== CLIP Similarity Evaluator - Corrected Version ===")
    
    try:
        # Initialize evaluator
        evaluator = CLIPSimilarityEvaluator(model_name="ViT-B/32")
        print("CLIP evaluator initialized successfully!")
        
        
        print("\n=== Evaluate All Characters ===")
        if os.path.exists("Characters"):
            print("Found Characters directory. Evaluating all characters...")
            try:
                all_results = evaluator.evaluate_all_characters_cut_images()
                if all_results:
                    total_chars = len([k for k, v in all_results.items() if v])
                    total_images = sum(len(v) for v in all_results.values())
                    print(f"Evaluation complete! Processed {total_chars} characters with {total_images} total images")
                    print("Check the generated JSON file for detailed results")
                else:
                    print("No images found to evaluate")
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
        else:
            print("Characters directory not found. Skipping all-characters evaluation.")
        
        # Example 3: Test with simple cases
        print("\n=== Simple Test Cases ===")
        test_cases = [
            ("a red image", "simple test - should work if you have a red image"),
            ("a photo of a cat", "test with cat image"),
            ("a cartoon character", "test with character image")
        ]
        
        for text, description in test_cases:
            print(f"Test case: {description}")
            print(f"Text: '{text}'")
            print("(Replace with actual image path to test)")
            print()
        
    except Exception as e:
        print(f"Error initializing CLIP evaluator: {str(e)}")


if __name__ == "__main__":
    main()