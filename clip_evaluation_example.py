#!/usr/bin/env python3
"""
CLIP Similarity Evaluation Example

This script demonstrates how to use the CLIP similarity evaluator to assess
the quality of generated images by measuring their semantic similarity to text prompts.

Usage examples:
1. Evaluate a single image: python clip_evaluation_example.py --image path/to/image.png --text "description"
2. Evaluate character images: python clip_evaluation_example.py --character Milo
3. Evaluate all characters: python clip_evaluation_example.py --all-characters
"""

import argparse
import sys
from pathlib import Path
import json

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from clip_similarity_evaluator import CLIPSimilarityEvaluator
except ImportError:
    print("Error: Could not import CLIPSimilarityEvaluator. Make sure you have installed the required dependencies:")
    print("pip install clip-by-openai")
    sys.exit(1)


def evaluate_single_image(image_path: str, description: str):
    """Evaluate a single image against its description."""
    print(f"=== Evaluating Single Image ===")
    print(f"Image: {image_path}")
    print(f"Description: {description}")
    
    evaluator = CLIPSimilarityEvaluator()
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    similarity = evaluator.calculate_similarity(image_path, description)
    print(f"Similarity Score: {similarity:.4f}")
    
    # Interpret the score
    if similarity >= 0.3:
        print("✅ Good alignment - image matches description well")
    elif similarity >= 0.2:
        print("⚠️ Moderate alignment - some semantic similarity")
    else:
        print("❌ Poor alignment - image doesn't match description well")


def evaluate_character(character_name: str):
    """Evaluate all generated images for a specific character."""
    print(f"=== Evaluating Character: {character_name} ===")
    
    evaluator = CLIPSimilarityEvaluator()
    results = evaluator.evaluate_character_generations(character_name)
    
    if not results:
        print(f"No images found for character: {character_name}")
        return
    
    # Print detailed results
    print(f"\nDetailed Results for {character_name}:")
    print("-" * 50)
    for image_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        status = "✅" if score >= 0.3 else "⚠️" if score >= 0.2 else "❌"
        print(f"{status} {image_name}: {score:.4f}")
    
    # Summary statistics
    scores = list(results.values())
    avg_score = sum(scores) / len(scores)
    high_quality_count = sum(1 for s in scores if s >= 0.3)
    
    print(f"\nSummary for {character_name}:")
    print(f"Average similarity: {avg_score:.4f}")
    print(f"High-quality images (≥0.3): {high_quality_count}/{len(scores)} ({high_quality_count/len(scores)*100:.1f}%)")


def evaluate_character_cut_images(character_name: str):
    """Evaluate cut images (character sheets) for a specific character."""
    print(f"=== Evaluating Character Cut Images: {character_name} ===")
    
    evaluator = CLIPSimilarityEvaluator()
    
    # Check if character exists
    character_dir = Path("Characters") / character_name
    if not character_dir.exists():
        print(f"Error: Character directory not found: {character_dir}")
        available_chars = [d.name for d in Path("Characters").iterdir() if d.is_dir()]
        print(f"Available characters: {', '.join(available_chars)}")
        return
    
    # Check if cut_images directory exists
    cut_images_dir = character_dir / "cut_images"
    if not cut_images_dir.exists():
        print(f"Error: Cut images directory not found: {cut_images_dir}")
        return
    
    # Evaluate cut images
    results = evaluator.evaluate_character_cut_images(character_name)
    
    if not results:
        print(f"No cut images found for {character_name}")
        return
    
    print(f"\nDetailed Results for {character_name} Cut Images:")
    print("-" * 50)
    
    # Sort results by similarity score (highest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for filename, similarity in sorted_results:
        if similarity >= 0.3:
            status = "✅"
        elif similarity >= 0.2:
            status = "⚠️"
        else:
            status = "❌"
        print(f"{status} {filename}: {similarity:.4f}")
    
    # Calculate summary statistics
    scores = list(results.values())
    avg_score = sum(scores) / len(scores)
    high_quality_count = sum(1 for score in scores if score >= 0.3)
    
    print(f"\nSummary for {character_name} Cut Images:")
    print(f"Average similarity: {avg_score:.4f}")
    print(f"High-quality images (≥0.3): {high_quality_count}/{len(results)} ({high_quality_count/len(results)*100:.1f}%)")

def evaluate_all_characters():
    """Evaluate all characters in the Characters directory."""
    print("=== Evaluating All Characters ===")
    
    characters_dir = Path("Characters")
    if not characters_dir.exists():
        print("Characters directory not found!")
        return
    
    character_folders = [d.name for d in characters_dir.iterdir() if d.is_dir()]
    
    if not character_folders:
        print("No character folders found!")
        return
    
    evaluator = CLIPSimilarityEvaluator()
    all_results = {}
    
    for character_name in character_folders:
        print(f"\nProcessing {character_name}...")
        results = evaluator.evaluate_character_generations(character_name)
        
        if results:
            scores = list(results.values())
            avg_score = sum(scores) / len(scores)
            high_quality_count = sum(1 for s in scores if s >= 0.3)
            
            all_results[character_name] = {
                'average_similarity': avg_score,
                'total_images': len(scores),
                'high_quality_images': high_quality_count,
                'high_quality_percentage': high_quality_count / len(scores) * 100
            }
            
            print(f"  Average similarity: {avg_score:.4f}")
            print(f"  High-quality images: {high_quality_count}/{len(scores)} ({high_quality_count/len(scores)*100:.1f}%)")
        else:
            print(f"  No images found for {character_name}")
    
    # Overall summary
    if all_results:
        print(f"\n=== Overall Summary ===")
        print("-" * 30)
        
        # Sort by average similarity
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['average_similarity'], reverse=True)
        
        for character_name, stats in sorted_results:
            status = "✅" if stats['average_similarity'] >= 0.3 else "⚠️" if stats['average_similarity'] >= 0.2 else "❌"
            print(f"{status} {character_name}: {stats['average_similarity']:.4f} "
                  f"({stats['high_quality_images']}/{stats['total_images']} high-quality)")
        
        # Save comprehensive results
        timestamp = Path("generated_images").glob("*/clip_similarity_evaluation_*.json")
        output_file = f"clip_evaluation_summary_{character_name}_{Path().name}.json"
        
        summary_data = {
            'evaluation_summary': 'All characters CLIP similarity evaluation',
            'total_characters': len(all_results),
            'character_results': all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate image-text similarity using CLIP")
    
    # Create mutually exclusive group for different evaluation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image to evaluate")
    group.add_argument("--character", help="Name of character to evaluate")
    group.add_argument("--character-cut", help="Name of character to evaluate cut images (character sheets)")
    group.add_argument("--all-characters", action="store_true", help="Evaluate all characters")
    
    # Additional arguments
    parser.add_argument("--text", help="Text description (required with --image)")
    parser.add_argument("--model", default="ViT-B-32", 
                       choices=["ViT-B-32", "ViT-B-16", "ViT-L-14"],
                       help="CLIP model to use (default: ViT-B-32)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image and not args.text:
        parser.error("--text is required when using --image")
    
    try:
        if args.image:
            evaluate_single_image(args.image, args.text)
        elif args.character:
            evaluate_character(args.character)
        elif args.character_cut:
            evaluate_character_cut_images(args.character_cut)
        elif args.all_characters:
            evaluate_all_characters()
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install clip-by-openai torch torchvision")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
