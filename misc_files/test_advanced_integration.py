#!/usr/bin/env python3
"""
Integration test for the advanced computer vision image cutting system.
This script tests the new advanced grid detection in the main pipeline.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def test_advanced_integration():
    """Test the advanced image processing integration."""
    print("=" * 60)
    print("ADVANCED IMAGE PROCESSING INTEGRATION TEST")
    print("=" * 60)
    
    # Test with existing Milo character data
    character_name = "Milo"
    print(f"\nTesting with character: {character_name}")
    
    # Check if character data exists
    characters_dir = Path("Characters")
    character_dir = characters_dir / character_name
    upscaled_dir = character_dir / "upscaled_images"
    
    if not character_dir.exists():
        print(f"✗ Character directory not found: {character_dir}")
        return False
    
    if not upscaled_dir.exists():
        print(f"✗ Upscaled images directory not found: {upscaled_dir}")
        return False
    
    # Check for required sheet files
    required_sheets = [
        f"Char_Sheet_{character_name}.png",
        f"Action_Sheet_{character_name}.png", 
        f"Emotion_Sheet_{character_name}.png"
    ]
    
    print(f"\nChecking for required sheet files in {upscaled_dir}:")
    missing_sheets = []
    for sheet in required_sheets:
        sheet_path = upscaled_dir / sheet
        if sheet_path.exists():
            print(f"  ✓ Found: {sheet}")
        else:
            print(f"  ✗ Missing: {sheet}")
            missing_sheets.append(sheet)
    
    if missing_sheets:
        print(f"\n✗ Cannot proceed - missing {len(missing_sheets)} sheet(s)")
        return False
    
    # Import and test the advanced processing function
    try:
        print(f"\nImporting advanced processing system...")
        from cut_an_image import process_character_images
        print("  ✓ Advanced system imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import advanced system: {e}")
        return False
    
    # Test the processing function
    print(f"\nRunning advanced image processing...")
    print("-" * 40)
    
    try:
        # Call the advanced processing function
        result = process_character_images(character_name, upscaled_dir)
        
        if result and result != False:
            print("-" * 40)
            print(f"✓ SUCCESS: Advanced processing completed!")
            print(f"✓ Output directory: {result}")
            
            # Check if output files were created
            if result.exists():
                output_files = list(result.glob("*.png"))
                print(f"✓ Generated {len(output_files)} cut image files:")
                for file in sorted(output_files)[:10]:  # Show first 10 files
                    print(f"    - {file.name}")
                if len(output_files) > 10:
                    print(f"    ... and {len(output_files) - 10} more files")
                
                return True
            else:
                print(f"✗ Output directory was not created: {result}")
                return False
        else:
            print("-" * 40)
            print(f"✗ FAILED: Advanced processing returned: {result}")
            return False
            
    except Exception as e:
        print("-" * 40)
        print(f"✗ EXCEPTION during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_main_pipeline_import():
    """Test that main.py can import the advanced system without errors."""
    print("\n" + "=" * 60)
    print("MAIN PIPELINE IMPORT TEST")
    print("=" * 60)
    
    try:
        print("Testing main.py import compatibility...")
        
        # Test the import that main.py now uses
        from cut_an_image import process_character_images
        print("  ✓ Import successful: process_character_images from cut_an_image")
        
        # Verify function signature
        import inspect
        sig = inspect.signature(process_character_images)
        params = list(sig.parameters.keys())
        expected_params = ['character_name', 'uncut_dir']
        
        if params == expected_params:
            print(f"  ✓ Function signature correct: {params}")
        else:
            print(f"  ✗ Function signature mismatch. Expected: {expected_params}, Got: {params}")
            return False
            
        print("  ✓ Main pipeline import compatibility confirmed")
        return True
        
    except Exception as e:
        print(f"  ✗ Import test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Advanced Image Processing Integration Tests...\n")
    
    # Test 1: Main pipeline import compatibility
    import_success = test_main_pipeline_import()
    
    # Test 2: Actual advanced processing 
    processing_success = test_advanced_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Import Compatibility Test: {'✓ PASS' if import_success else '✗ FAIL'}")
    print(f"Advanced Processing Test: {'✓ PASS' if processing_success else '✗ FAIL'}")
    
    if import_success and processing_success:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("✓ Advanced computer vision system successfully integrated")
        print("✓ Main pipeline ready to use enhanced grid detection")
        print("✓ The pipeline will now use sophisticated edge detection and line clustering")
    else:
        print("\n❌ SOME TESTS FAILED ❌")
        print("Please review the errors above and fix before proceeding")
    
    print("=" * 60)
