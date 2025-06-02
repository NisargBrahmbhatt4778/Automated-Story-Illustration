"""
Advanced Image Grid Detection and Cutting System

This module provides sophisticated computer vision-based grid detection for cutting
character sheets, action sheets, and emotion sheets into individual cells.

Features:
- Adaptive thresholding for robust grid line detection
- Canny edge detection with configurable parameters
- Hough line transform for line segment detection
- Intelligent line clustering and filtering
- Comprehensive debug output and logging
- Fallback to simple grid cutting if detection fails
- Support for multiple sheet types and dimensions

"""

import cv2
import numpy as np
import os
import math
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(output_dir):
    """
    Sets up logging for the image processing operations.
    
    Args:
        output_dir (str): Directory to save log files
    
    Returns:
        logging.Logger: Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('GridDetection')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create file handler
    log_file = os.path.join(output_dir, f"grid_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_sheet_parameters(sheet_type="character", image_width=1536, image_height=1024):
    """
    Get optimized parameters for different sheet types.
    
    Args:
        sheet_type (str): Type of sheet ("character", "action", "emotion")
        image_width (int): Width of the input image
        image_height (int): Height of the input image
    
    Returns:
        dict: Dictionary containing all processing parameters
    """
    base_params = {
        "blur_kernel_size": (3, 3),
        "hough_rho": 1,
        "hough_theta": np.pi / 180,
    }
    
    if sheet_type == "emotion":
        # Parameters optimized for 3x3 emotion grids (typically 1024x1024)
        params = {
            **base_params,
            "adaptive_thresh_block_size": 41,
            "adaptive_thresh_C": 15,
            "canny_low_thresh": 70,
            "canny_high_thresh": 200,
            "hough_threshold": 250,
            "hough_max_line_gap": 30,
            "max_angle_deviation_deg": 3.0,
            "line_cluster_separation": 30,
            "min_cell_dimension": 50,
            "expected_grid": (3, 3)
        }
    else:
        # Parameters optimized for 2x3 character/action grids (typically 1536x1024)
        params = {
            **base_params,
            "adaptive_thresh_block_size": 51,
            "adaptive_thresh_C": 18,
            "canny_low_thresh": 80,
            "canny_high_thresh": 220,
            "hough_threshold": 300,
            "hough_max_line_gap": 35,
            "max_angle_deviation_deg": 3.5,
            "line_cluster_separation": 35,
            "min_cell_dimension": 60,
            "expected_grid": (2, 3)
        }
    
    # Calculate dynamic min line length
    params["hough_min_line_length"] = max(
        int(min(image_width, image_height) * 0.35),
        params["min_cell_dimension"] * 2
    )
    
    return params

def _process_line_coords(coords, min_cell_dimension, image_boundary_val, cluster_min_separation=30):
    """
    Processes a list of 1D coordinates (x or y) to find distinct grid lines.
    It sorts, clusters nearby coordinates, and ensures a minimum cell dimension.
    
    Args:
        coords (list): List of coordinate values
        min_cell_dimension (int): Minimum size for a cell
        image_boundary_val (int): Maximum boundary value (width or height)
        cluster_min_separation (int): Minimum separation for clustering
    
    Returns:
        list: Processed and filtered grid line coordinates
    """
    if not coords: 
        return sorted(list(set([0, image_boundary_val])))

    valid_coords = [int(round(c)) for c in coords if 0 <= c <= image_boundary_val]
    if not valid_coords:
        return sorted(list(set([0, image_boundary_val])))
        
    sorted_coords = sorted(list(set(valid_coords)))

    clustered_lines = []
    if sorted_coords: 
        current_cluster = [sorted_coords[0]]
        for i in range(1, len(sorted_coords)):
            coord = sorted_coords[i]
            if current_cluster and abs(coord - np.mean(current_cluster)) < cluster_min_separation:
                current_cluster.append(coord)
            else:
                if current_cluster: 
                    clustered_lines.append(int(round(np.mean(current_cluster))))
                current_cluster = [coord] 
        if current_cluster: 
            clustered_lines.append(int(round(np.mean(current_cluster))))

    final_grid_lines = [0]
    unique_clustered_lines = sorted(list(set(clustered_lines)))

    for line_pos in unique_clustered_lines:
        if line_pos <= 0: 
            continue
        if line_pos >= image_boundary_val: 
            continue
        if final_grid_lines and line_pos - final_grid_lines[-1] >= min_cell_dimension:
            final_grid_lines.append(line_pos)
        elif not final_grid_lines and line_pos >= min_cell_dimension : 
             final_grid_lines.append(line_pos)

    if not final_grid_lines or image_boundary_val - final_grid_lines[-1] >= min_cell_dimension:
        if image_boundary_val not in final_grid_lines: 
            if final_grid_lines and image_boundary_val - final_grid_lines[-1] >= min_cell_dimension:
                 final_grid_lines.append(image_boundary_val)
            elif not final_grid_lines and image_boundary_val >=min_cell_dimension: 
                 final_grid_lines.append(image_boundary_val)

    elif len(final_grid_lines) > 1 and image_boundary_val > final_grid_lines[-1] : 
        if final_grid_lines[-2] < image_boundary_val and \
           image_boundary_val - final_grid_lines[-2] >= min_cell_dimension: 
             final_grid_lines[-1] = image_boundary_val
    elif final_grid_lines == [0] and image_boundary_val >= min_cell_dimension : 
        final_grid_lines.append(image_boundary_val)
    
    if final_grid_lines == [0] and image_boundary_val == 0: 
        return []
        
    return sorted(list(set(final_grid_lines)))

def simple_grid_fallback(image_path, output_dir, sheet_type="character", logger=None):
    """
    Fallback method using simple grid cutting when computer vision detection fails.
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory to save cut cells
        sheet_type (str): Type of sheet for grid configuration
        logger (logging.Logger): Logger instance for output
    
    Returns:
        int: Number of cells successfully extracted
    """
    if logger:
        logger.info(f"Using simple grid fallback for {sheet_type} sheet")
    
    try:
        import cv2
        original_image = cv2.imread(image_path)
        if original_image is None:
            if logger:
                logger.error(f"Could not load image: {image_path}")
            return 0
        
        img_height, img_width = original_image.shape[:2]
        
        if sheet_type == "emotion":
            # 3x3 grid for emotion sheets
            rows, cols = 3, 3
            section_width = img_width // cols
            section_height = img_height // rows
        else:
            # 2x3 grid for character/action sheets
            rows, cols = 2, 3
            section_width = img_width // cols
            section_height = img_height // rows
        
        cell_count = 0
        for row in range(rows):
            for col in range(cols):
                y_start = row * section_height
                y_end = (row + 1) * section_height
                x_start = col * section_width
                x_end = (col + 1) * section_width
                
                cell = original_image[y_start:y_end, x_start:x_end]
                if cell.size == 0:
                    continue
                
                # Generate filename based on sheet type
                if sheet_type == "character":
                    cell_filename = os.path.join(output_dir, f"char_sheet_{cell_count + 1}.png")
                elif sheet_type == "action":
                    cell_filename = os.path.join(output_dir, f"action_sheet_{cell_count + 1}.png")
                else:  # emotion
                    cell_filename = os.path.join(output_dir, f"emotion_sheet_{cell_count + 1}.png")
                
                cv2.imwrite(cell_filename, cell)
                cell_count += 1
        
        if logger:
            logger.info(f"Simple fallback extracted {cell_count} cells")
        
        return cell_count
        
    except Exception as e:
        if logger:
            logger.error(f"Simple fallback failed: {str(e)}")
        return 0

def advanced_grid_detection(image_path, output_dir="output_cells_advanced", sheet_type="character", logger=None):
    """
    Advanced grid detection and cutting using computer vision techniques.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save cut cells and debug images
        sheet_type (str): Type of sheet ("character", "action", "emotion")
        logger (logging.Logger): Logger instance for output
    
    Returns:
        int: Number of cells successfully extracted
    """
    if logger is None:
        logger = setup_logging(output_dir)
    
    logger.info(f"Starting advanced grid detection for {sheet_type} sheet")
    logger.info(f"Input image: {image_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and validate image
    original_image = cv2.imread(image_path)
    if original_image is None:
        logger.error(f"Could not load image from {image_path}")
        return 0
    
    img_height, img_width = original_image.shape[:2]
    logger.info(f"Image dimensions: {img_width}x{img_height}")
    
    # Get optimized parameters for this sheet type
    params = get_sheet_parameters(sheet_type, img_width, img_height)
    
    logger.info("=== Computer Vision Parameters ===")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 35)

    
    # Image processing pipeline
    logger.info("Starting image processing pipeline...")
    
    # Step 1: Convert to grayscale and blur
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, params["blur_kernel_size"], 0)
    
    # Step 2: Adaptive thresholding
    logger.info("Applying adaptive thresholding...")
    adaptive_th_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, params["adaptive_thresh_block_size"], params["adaptive_thresh_C"]
    )
    cv2.imwrite(os.path.join(output_dir, f"debug_adaptive_thresh_{sheet_type}.png"), adaptive_th_image)

    # Step 3: Edge detection
    logger.info("Performing Canny edge detection...")
    edges = cv2.Canny(adaptive_th_image, params["canny_low_thresh"], params["canny_high_thresh"])
    cv2.imwrite(os.path.join(output_dir, f"debug_edges_{sheet_type}.png"), edges)

    # Step 4: Line detection using Hough transform
    logger.info("Detecting lines with Hough transform...")
    lines = cv2.HoughLinesP(
        edges, params["hough_rho"], params["hough_theta"], params["hough_threshold"],
        minLineLength=params["hough_min_line_length"],
        maxLineGap=params["hough_max_line_gap"]
    )

    if lines is None:
        logger.warning("No line segments detected by Hough transform. Using fallback method.")
        return simple_grid_fallback(image_path, output_dir, sheet_type, logger)

    # Step 5: Process detected lines
    logger.info(f"Found {len(lines)} raw line segments")
    horizontal_coords = []
    vertical_coords = []
    max_angle_dev_rad = np.deg2rad(params["max_angle_deviation_deg"])
    
    # Create debug image showing all detected lines
    debug_image_lines = original_image.copy()

    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        cv2.line(debug_image_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        delta_x = x2 - x1
        delta_y = y2 - y1
        
        if delta_x == 0:
            angle_rad = np.pi / 2
        else:
            angle_rad = np.arctan(abs(delta_y) / abs(delta_x))

        if angle_rad < max_angle_dev_rad:
            horizontal_coords.append((y1 + y2) / 2)
        elif abs(angle_rad - np.pi/2) < max_angle_dev_rad:
            vertical_coords.append((x1 + x2) / 2)
            
    cv2.imwrite(os.path.join(output_dir, f"debug_detected_lines_{sheet_type}.png"), debug_image_lines)

    logger.info(f"Raw horizontal line candidates: {len(horizontal_coords)}")
    logger.info(f"Raw vertical line candidates: {len(vertical_coords)}")

    # Step 6: Process and cluster grid lines
    processed_horz_lines = _process_line_coords(
        horizontal_coords, params["min_cell_dimension"], img_height, params["line_cluster_separation"]
    )
    processed_vert_lines = _process_line_coords(
        vertical_coords, params["min_cell_dimension"], img_width, params["line_cluster_separation"]
    )

    logger.info(f"Processed horizontal grid lines: {processed_horz_lines}")
    logger.info(f"Processed vertical grid lines: {processed_vert_lines}")

    # Create debug image showing processed grid lines
    debug_processed_lines_image = original_image.copy()
    for y_coord in processed_horz_lines:
        cv2.line(debug_processed_lines_image, (0, int(y_coord)), (img_width, int(y_coord)), (0, 255, 255), 3)
    for x_coord in processed_vert_lines:
        cv2.line(debug_processed_lines_image, (int(x_coord), 0), (int(x_coord), img_height), (255, 0, 255), 3)
    cv2.imwrite(os.path.join(output_dir, f"debug_processed_lines_{sheet_type}.png"), debug_processed_lines_image)


    # Step 7: Extract cells from detected grid
    cell_count = 0
    if len(processed_horz_lines) < 2 or len(processed_vert_lines) < 2:
        logger.warning("Not enough distinct grid lines found. Using fallback method.")
        return simple_grid_fallback(image_path, output_dir, sheet_type, logger)

    logger.info("Extracting cells from detected grid...")
    for i in range(len(processed_horz_lines) - 1):
        y_start = int(processed_horz_lines[i])
        y_end = int(processed_horz_lines[i+1])
        if y_end <= y_start:
            continue

        for j in range(len(processed_vert_lines) - 1):
            x_start = int(processed_vert_lines[j])
            x_end = int(processed_vert_lines[j+1])
            if x_end <= x_start:
                continue

            if (y_end - y_start >= params["min_cell_dimension"]) and \
               (x_end - x_start >= params["min_cell_dimension"]):
                cell = original_image[y_start:y_end, x_start:x_end]
                if cell.size == 0:
                    continue
                
                # Generate filename based on sheet type
                if sheet_type == "character":
                    cell_filename = os.path.join(output_dir, f"char_sheet_{cell_count + 1}.png")
                elif sheet_type == "action":
                    cell_filename = os.path.join(output_dir, f"action_sheet_{cell_count + 1}.png")
                else:  # emotion
                    cell_filename = os.path.join(output_dir, f"emotion_sheet_{cell_count + 1}.png")
                
                cv2.imwrite(cell_filename, cell)
                cell_count += 1
                logger.info(f"Extracted cell {cell_count}: {os.path.basename(cell_filename)}")

    logger.info(f"Successfully extracted {cell_count} cells using advanced detection")
    
    # Validate against expected grid
    expected_cells = params["expected_grid"][0] * params["expected_grid"][1]
    if cell_count != expected_cells:
        logger.warning(f"Expected {expected_cells} cells for {sheet_type} sheet, but got {cell_count}")
    
    return cell_count

def cut_character_sheet(image_path, output_dir, character_name=None):
    """
    Cut a character sheet using advanced grid detection.
    
    Args:
        image_path (str): Path to the character sheet image
        output_dir (str): Directory to save cut images
        character_name (str): Name of character (for logging)
    
    Returns:
        int: Number of cells extracted
    """
    debug_dir = os.path.join(output_dir, "debug_character")
    return advanced_grid_detection(str(image_path), debug_dir, "character")

def cut_action_sheet(image_path, output_dir, character_name=None):
    """
    Cut an action sheet using advanced grid detection.
    
    Args:
        image_path (str): Path to the action sheet image
        output_dir (str): Directory to save cut images
        character_name (str): Name of character (for logging)
    
    Returns:
        int: Number of cells extracted
    """
    debug_dir = os.path.join(output_dir, "debug_action")
    return advanced_grid_detection(str(image_path), debug_dir, "action")

def cut_emotion_sheet(image_path, output_dir, character_name=None):
    """
    Cut an emotion sheet using advanced grid detection.
    
    Args:
        image_path (str): Path to the emotion sheet image
        output_dir (str): Directory to save cut images
        character_name (str): Name of character (for logging)
    
    Returns:
        int: Number of cells extracted
    """
    debug_dir = os.path.join(output_dir, "debug_emotion")
    return advanced_grid_detection(str(image_path), debug_dir, "emotion")

def process_character_images(character_name, uncut_dir):
    """
    Processes all character images using advanced computer vision grid detection.
    This function maintains compatibility with the simple image processor interface
    while providing enhanced grid detection capabilities.
    
    Args:
        character_name (str): Name of the character
        uncut_dir (Path): Path to the uncut_images directory (usually upscaled)
        
    Returns:
        Path: Path to cut images directory on success, False on failure
    """
    from pathlib import Path
    
    print("\n=== Advanced Image Processing System ===")
    print("Using computer vision-based grid detection")
    
    # Create cut_images directory
    cut_dir = uncut_dir.parent / 'cut_images'
    cut_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for upscaled images (prioritize upscaled over original)
    upscaled_dir = Path(f"characters/{character_name}/upscaled_images")
    use_upscaled = upscaled_dir.exists() and any(upscaled_dir.glob("*.png"))
    
    if use_upscaled:
        print("  → Using upscaled images for advanced processing...")
        input_dir = upscaled_dir
    else:
        print("  → Using original images for advanced processing...")
        input_dir = uncut_dir
    
    # Set up logging
    logger = setup_logging(str(cut_dir))
    logger.info(f"Starting advanced processing for character: {character_name}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {cut_dir}")
    
    success_count = 0
    total_sheets = 0
    
    # Process character sheet
    char_sheet = input_dir / f"Char_Sheet_{character_name}.png"
    if char_sheet.exists():
        total_sheets += 1
        print("\n  → Processing character sheet with advanced grid detection...")
        try:
            cell_count = cut_character_sheet(char_sheet, str(cut_dir), character_name)
            if cell_count > 0:
                success_count += 1
                print(f"    ✓ Character sheet processed: {cell_count} cells extracted")
                logger.info(f"Character sheet success: {cell_count} cells")
            else:
                print(f"    ✗ Character sheet processing failed")
                logger.error(f"Character sheet failed: 0 cells extracted")
        except Exception as e:
            print(f"    ✗ Character sheet error: {str(e)}")
            logger.error(f"Character sheet exception: {str(e)}")
    else:
        print(f"    ! Character sheet not found at {char_sheet}")
        logger.warning(f"Character sheet not found: {char_sheet}")
    
    # Process action sheet
    action_sheet = input_dir / f"Action_Sheet_{character_name}.png"
    if action_sheet.exists():
        total_sheets += 1
        print("\n  → Processing action sheet with advanced grid detection...")
        try:
            cell_count = cut_action_sheet(action_sheet, str(cut_dir), character_name)
            if cell_count > 0:
                success_count += 1
                print(f"    ✓ Action sheet processed: {cell_count} cells extracted")
                logger.info(f"Action sheet success: {cell_count} cells")
            else:
                print(f"    ✗ Action sheet processing failed")
                logger.error(f"Action sheet failed: 0 cells extracted")
        except Exception as e:
            print(f"    ✗ Action sheet error: {str(e)}")
            logger.error(f"Action sheet exception: {str(e)}")
    else:
        print(f"    ! Action sheet not found at {action_sheet}")
        logger.warning(f"Action sheet not found: {action_sheet}")
    
    # Process emotion sheet
    emotion_sheet = input_dir / f"Emotion_Sheet_{character_name}.png"
    if emotion_sheet.exists():
        total_sheets += 1
        print("\n  → Processing emotion sheet with advanced grid detection...")
        try:
            cell_count = cut_emotion_sheet(emotion_sheet, str(cut_dir), character_name)
            if cell_count > 0:
                success_count += 1
                print(f"    ✓ Emotion sheet processed: {cell_count} cells extracted")
                logger.info(f"Emotion sheet success: {cell_count} cells")
            else:
                print(f"    ✗ Emotion sheet processing failed")
                logger.error(f"Emotion sheet failed: 0 cells extracted")
        except Exception as e:
            print(f"    ✗ Emotion sheet error: {str(e)}")
            logger.error(f"Emotion sheet exception: {str(e)}")
    else:
        print(f"    ! Emotion sheet not found at {emotion_sheet}")
        logger.warning(f"Emotion sheet not found: {emotion_sheet}")
    
    # Summary
    print(f"\n=== Advanced Processing Summary ===")
    print(f"  Sheets found: {total_sheets}")
    print(f"  Sheets processed successfully: {success_count}")
    
    if total_sheets == 0:
        print("  ✗ No character sheets found!")
        logger.error("No sheets found for processing")
        return False
    elif success_count == 0:
        print("  ✗ All sheet processing failed!")
        logger.error("All sheet processing failed")
        return False
    elif success_count < total_sheets:
        print(f"  ⚠ Partial success: {success_count}/{total_sheets} sheets processed")
        logger.warning(f"Partial success: {success_count}/{total_sheets}")
    else:
        print(f"  ✓ All sheets processed successfully!")
        logger.info("All sheets processed successfully")
    
    print(f"  Cut images saved to: {cut_dir}")
    logger.info(f"Processing complete. Output directory: {cut_dir}")
    
    return cut_dir

# Legacy function name for backward compatibility
def split_image_into_cells(image_path, output_dir="output_cells_advanced", sheet_type="character"):
    """
    Legacy function name - now uses advanced grid detection.
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Output directory
        sheet_type (str): Type of sheet
    
    Returns:
        int: Number of cells extracted
    """
    return advanced_grid_detection(image_path, output_dir, sheet_type)

if __name__ == "__main__":
    """
    Main execution section with enhanced functionality and better error handling.
    """
    print("=" * 60)
    print("Advanced Image Grid Detection and Cutting System")
    print("=" * 60)
    
    # Define possible image paths
    test_images = {
        "hedgehog": "ChatGPT Image May 28, 2025 at 03_27_19 AM.jpg",
        "warrior": "ChatGPT Image May 28, 2025 at 03_33_53 AM.jpg",
        "sample": "z_GPT_Templates/sample2.png"
    }
    
    # Try to find an available test image
    image_to_process = None
    sheet_type_to_use = "character"  # Default
    
    # Check for specific test images
    for name, path in test_images.items():
        if os.path.exists(path):
            image_to_process = path
            print(f"Found test image: {name} ({path})")
            break
    
    # If no test images found, create a dummy image
    if not image_to_process:
        print("No test images found. Creating dummy image for demonstration...")
        dummy_image_path = "z_GPT_Templates/dummy_grid.png"
        os.makedirs("z_GPT_Templates", exist_ok=True)
        
        # Create a dummy 3x3 grid image for testing
        img_h, img_w = 600, 600
        dummy = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240
        
        # Draw grid lines
        line_color = (50, 50, 50)
        line_thickness = 2
        
        # Vertical lines
        for i in range(1, 3):
            x = i * img_w // 3
            cv2.line(dummy, (x, 0), (x, img_h), line_color, line_thickness)
        
        # Horizontal lines
        for i in range(1, 3):
            y = i * img_h // 3
            cv2.line(dummy, (0, y), (img_w, y), line_color, line_thickness)
        
        # Add some content to cells
        font = cv2.FONT_HERSHEY_SIMPLEX
        for row in range(3):
            for col in range(3):
                cell_num = row * 3 + col + 1
                x = col * img_w // 3 + img_w // 6
                y = row * img_h // 3 + img_h // 6
                cv2.putText(dummy, f"Cell {cell_num}", (x-40, y), font, 0.8, (100, 100, 100), 2)
        
        cv2.imwrite(dummy_image_path, dummy)
        image_to_process = dummy_image_path
        sheet_type_to_use = "emotion"  # 3x3 grid
        print(f"Created dummy image: {dummy_image_path}")
    
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = f"output_cells_advanced_{timestamp}"
    
    print(f"\nProcessing Configuration:")
    print(f"  Input Image: {image_to_process}")
    print(f"  Sheet Type: {sheet_type_to_use}")
    print(f"  Output Directory: {output_directory}")
    print(f"  Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        # Process the image
        cell_count = advanced_grid_detection(
            image_path=image_to_process,
            output_dir=output_directory,
            sheet_type=sheet_type_to_use
        )
        
        if cell_count > 0:
            print(f"\n✓ SUCCESS: Extracted {cell_count} cells successfully!")
            print(f"✓ Results saved to: {output_directory}")
            print(f"✓ Debug images available in the output directory")
        else:
            print(f"\n✗ FAILED: No cells were extracted from the image")
            print(f"✗ Check debug images in: {output_directory}")
            
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {str(e)}")
        print(f"✗ Please check the input image and try again")
    
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)

