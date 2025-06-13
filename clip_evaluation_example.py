#!/usr/bin/env python3
"""
CLIP Similarity Evaluation Example - Corrected Version

This script demonstrates how to use the corrected CLIP similarity evaluator to assess
the quality of generated images by measuring their semantic similarity to text prompts.

"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.clip_similarity_evaluator import CLIPSimilarityEvaluator
except ImportError:
    print("Error: Could not import CLIPSimilarityEvaluator.")
    sys.exit(1)


def evaluate_single_image(image_path: str, description: str, debug: bool = False):
    """Evaluate a single image against its description."""
    print(f"=== Evaluating Single Image ===")
    print(f"Image: {image_path}")
    print(f"Description: {description}")
    
    try:
        evaluator = CLIPSimilarityEvaluator(model_name="ViT-B/32")
    except Exception as e:
        print(f"Failed to initialize CLIP evaluator: {str(e)}")
        return
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    if debug:
        # Use debug evaluation for detailed output
        debug_info = evaluator.debug_single_evaluation(image_path, description)
        
        if "error" in debug_info:
            print(f"Debug evaluation failed: {debug_info['error']}")
            return
        
        print(f"\n=== Debug Results ===")
        print(f"Similarity Score: {debug_info['similarity']:.6f}")
        print(f"Interpretation: {debug_info['interpretation']}")
        print(f"Image size: {debug_info['image_size']}")
        print(f"Model used: {debug_info['model']}")
        print(f"Device: {debug_info['device']}")
        print(f"Image features norm: {debug_info['image_features_norm']:.6f}")
        print(f"Text features norm: {debug_info['text_features_norm']:.6f}")
        
    else:
        # Regular evaluation
        similarity = evaluator.calculate_similarity(image_path, description)
        print(f"\nSimilarity Score: {similarity:.6f}")
    
        # Interpret the score
        if similarity >= 0.3:
            print("Excellent alignment - image matches description very well")
        elif similarity >= 0.2:
            print("Good alignment - image has strong semantic similarity")
        elif similarity >= 0.1:
            print("Fair alignment - image has some semantic similarity")
        else:
            print("Poor alignment - image doesn't match description well")
    


def evaluate_character(character_name: str):
    """Evaluate all generated images for a specific character."""
    print(f"=== Evaluating Character: {character_name} ===")
    
    try:
        evaluator = CLIPSimilarityEvaluator(model_name="ViT-B/32")
    except Exception as e:
        print(f"Failed to initialize CLIP evaluator: {str(e)}")
        return
    
    results = evaluator.evaluate_character_generations(character_name)
    
    if not results:
        print(f"No images found for character: {character_name}")
        
        # Check if character directory exists
        character_dir = Path("generated_images") / character_name
        if not character_dir.exists():
            print(f"Character directory not found: {character_dir}")
            
            # Show available characters
            base_dir = Path("generated_images")
            if base_dir.exists():
                available_chars = [d.name for d in base_dir.iterdir() if d.is_dir()]
                if available_chars:
                    print(f"Available characters: {', '.join(available_chars)}")
                else:
                    print("No character directories found in generated_images/")
            else:
                print("generated_images/ directory not found")
        return
    
    # Print detailed results
    print(f"\n=== Detailed Results for {character_name} ===")
    print("-" * 60)
    
    # Sort results by similarity score (highest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for image_name, score in sorted_results:
        if score >= 0.3:
            status = "Excellent"
        elif score >= 0.2:
            status = "Good     "
        elif score >= 0.1:
            status = "Fair     "
        else:
            status = "Poor     "
        print(f"{status} - {image_name}: {score:.6f}")
    
    # Summary statistics
    scores = list(results.values())
    avg_score = sum(scores) / len(scores)
    excellent_count = sum(1 for s in scores if s >= 0.3)
    good_count = sum(1 for s in scores if 0.2 <= s < 0.3)
    fair_count = sum(1 for s in scores if 0.1 <= s < 0.2)
    poor_count = sum(1 for s in scores if s < 0.1)
    
    print(f"\n=== Summary for {character_name} ===")
    print(f"Average similarity: {avg_score:.6f}")
    print(f"Total images: {len(scores)}")
    print(f"Quality distribution:")
    print(f"  Excellent (≥0.30): {excellent_count}/{len(scores)} ({excellent_count/len(scores)*100:.1f}%)")
    print(f"  Good (0.20-0.29):  {good_count}/{len(scores)} ({good_count/len(scores)*100:.1f}%)")
    print(f"  Fair (0.10-0.19):  {fair_count}/{len(scores)} ({fair_count/len(scores)*100:.1f}%)")
    print(f"  Poor (<0.10):      {poor_count}/{len(scores)} ({poor_count/len(scores)*100:.1f}%)")


def evaluate_all_characters():
    """Evaluate all characters in the generated_images directory."""
    print("=== Evaluating All Characters ===")
    
    base_dir = Path("generated_images")
    if not base_dir.exists():
        print("generated_images directory not found!")
        return
    
    character_folders = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    if not character_folders:
        print("No character folders found in generated_images/!")
        return
    
    print(f"Found {len(character_folders)} characters: {', '.join(character_folders)}")
    
    try:
        evaluator = CLIPSimilarityEvaluator(model_name="ViT-B/32")
    except Exception as e:
        print(f"Failed to initialize CLIP evaluator: {str(e)}")
        return
    
    all_results = {}
    overall_scores = []
    
    for character_name in character_folders:
        print(f"\n--- Processing {character_name} ---")
        results = evaluator.evaluate_character_generations(character_name)
        
        if results:
            scores = list(results.values())
            avg_score = sum(scores) / len(scores)
            excellent_count = sum(1 for s in scores if s >= 0.3)
            good_count = sum(1 for s in scores if 0.2 <= s < 0.3)
            fair_count = sum(1 for s in scores if 0.1 <= s < 0.2)
            poor_count = sum(1 for s in scores if s < 0.1)
            
            all_results[character_name] = {
                'average_similarity': avg_score,
                'total_images': len(scores),
                'excellent_images': excellent_count,
                'good_images': good_count,
                'fair_images': fair_count,
                'poor_images': poor_count,
                'excellent_percentage': excellent_count / len(scores) * 100,
                'individual_scores': results
            }
            
            overall_scores.extend(scores)
            
            print(f"  Average similarity: {avg_score:.6f}")
            print(f"  Quality: {excellent_count}E {good_count}G {fair_count}F {poor_count}P (out of {len(scores)})")
        else:
            print(f"  No images found for {character_name}")
    
    # Overall summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"=== OVERALL SUMMARY ===")
        print(f"{'='*60}")
        
        # Sort characters by average similarity
        sorted_characters = sorted(all_results.items(), key=lambda x: x[1]['average_similarity'], reverse=True)
        
        print(f"\n=== Character Rankings ===")
        for i, (character_name, stats) in enumerate(sorted_characters, 1):
            status = "EXCELLENT" if stats['average_similarity'] >= 0.3 else "GOOD" if stats['average_similarity'] >= 0.2 else "POOR"
            print(f"{i:2d}. {status:9s} {character_name:15s}: {stats['average_similarity']:.6f} "
                  f"({stats['excellent_images']}/{stats['total_images']} excellent)")
        
        # Overall statistics
        if overall_scores:
            overall_avg = sum(overall_scores) / len(overall_scores)
            overall_excellent = sum(1 for s in overall_scores if s >= 0.3)
            overall_good = sum(1 for s in overall_scores if 0.2 <= s < 0.3)
            overall_fair = sum(1 for s in overall_scores if 0.1 <= s < 0.2)
            overall_poor = sum(1 for s in overall_scores if s < 0.1)
            
            print(f"\n=== Project-Wide Statistics ===")
            print(f"Total characters evaluated: {len(all_results)}")
            print(f"Total images evaluated: {len(overall_scores)}")
            print(f"Overall average similarity: {overall_avg:.6f}")
            print(f"Overall quality distribution:")
            print(f"  Excellent (≥0.30): {overall_excellent}/{len(overall_scores)} ({overall_excellent/len(overall_scores)*100:.1f}%)")
            print(f"  Good (0.20-0.29):  {overall_good}/{len(overall_scores)} ({overall_good/len(overall_scores)*100:.1f}%)")
            print(f"  Fair (0.10-0.19):  {overall_fair}/{len(overall_scores)} ({overall_fair/len(overall_scores)*100:.1f}%)")
            print(f"  Poor (<0.10):      {overall_poor}/{len(overall_scores)} ({overall_poor/len(overall_scores)*100:.1f}%)")
        
        # Save comprehensive results
        output_file = f"clip_evaluation_summary_{len(all_results)}chars.json"
        summary_data = {
            'evaluation_date': Path().absolute().name,
            'evaluation_summary': f'All characters CLIP similarity evaluation ({len(all_results)} characters)',
            'total_characters': len(all_results),
            'total_images': len(overall_scores),
            'overall_average_similarity': overall_avg if overall_scores else 0,
            'character_results': all_results
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"\nDetailed results saved to: {output_file}")
        except Exception as e:
            print(f"Could not save results file: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate image-text similarity using corrected CLIP implementation")
    
    # Create mutually exclusive group for different evaluation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image to evaluate")
    group.add_argument("--character", help="Name of character to evaluate")
    group.add_argument("--all-characters", action="store_true", help="Evaluate all characters")
    
    # Additional arguments
    parser.add_argument("--text", help="Text description (required with --image)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed evaluation info")
    parser.add_argument("--model", default="ViT-B/32", 
                       choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", 
                               "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
                       help="CLIP model to use (default: ViT-B/32)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image and not args.text:
        parser.error("--text is required when using --image")
    
    # Print header
    print("="*60)
    print("CLIP Similarity Evaluator - Corrected Version")
    print("="*60)
    
    try:
        if args.image:
            evaluate_single_image(args.image, args.text, debug=args.debug)
        elif args.character:
            evaluate_character(args.character)
        elif args.all_characters:
            evaluate_all_characters()
    
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed the original CLIP:")
        print("   pip install git+https://github.com/openai/CLIP.git")
        print("2. Install other dependencies:")
        print("   pip install torch torchvision Pillow numpy")
        print("3. Check that image paths exist and are accessible")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())