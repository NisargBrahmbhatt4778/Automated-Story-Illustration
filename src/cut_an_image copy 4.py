import cv2
import numpy as np
import os
import math

def _process_line_coords(coords, min_cell_dimension, image_boundary_val, cluster_min_separation=25): # Increased cluster_sep
    """
    Processes a list of 1D coordinates (x or y) to find distinct grid lines.
    It sorts, clusters nearby coordinates, and ensures a minimum cell dimension.
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


def split_image_into_cells(image_path, output_dir="output_cells_tuned_v3"): # Changed output dir
    """
    Detects grid lines in an image using adaptive thresholding and saves individual cells.
    Parameters tuned to be very selective for complex images.
    """
    # --- Parameters (CRITICAL: TUNE THESE BASED ON DEBUG OUTPUT FOR WARRIOR IMAGE) ---
    # Gaussian Blur
    blur_kernel_size = (3, 3) 
    # Adaptive Thresholding (Goal: Make grid lines white, rest black for Canny if using THRESH_BINARY_INV)
    adaptive_thresh_block_size = 41 # Odd number. Significantly Increased. Try 31, 41, 51...
    adaptive_thresh_C = 15       # Constant subtracted. Significantly Increased. Try values from 15 to 25.
    # Canny Edge Detection
    canny_low_thresh = 70       # Increased low threshold
    canny_high_thresh = 200     # Increased high threshold
    # Hough Line Transform (HoughLinesP)
    hough_rho = 1  
    hough_theta = np.pi / 180  
    hough_threshold = 250       # CRITICAL: Min votes. Massively Increased. Try 200, 250, 300...
    # hough_min_line_length is dynamic
    hough_max_line_gap = 30     # Max gap.
    # Line classification
    max_angle_deviation_deg = 4.0 # Slightly stricter angle
    # Line processing (for _process_line_coords)
    line_cluster_separation = 30 # Cluster lines within this pixel distance. Increased.
    min_cell_dimension = 50      # Minimum width/height for a cell. Increased.

    # --- End Parameters ---

    print("--- Using Parameters (Tuned v3 - Aggressive) ---")
    print(f"  Blur Kernel Size: {blur_kernel_size}")
    print(f"  Adaptive Thresh Block Size: {adaptive_thresh_block_size}")
    print(f"  Adaptive Thresh C: {adaptive_thresh_C}")
    print(f"  Canny Low: {canny_low_thresh}, Canny High: {canny_high_thresh}")
    print(f"  Hough Rho: {hough_rho}, Theta: Rads")
    print(f"  Hough Threshold: {hough_threshold}")
    print(f"  Hough Max Line Gap: {hough_max_line_gap}")
    print(f"  Max Angle Deviation (Deg): {max_angle_deviation_deg}")
    print(f"  Line Cluster Separation: {line_cluster_separation}")
    print(f"  Min Cell Dimension: {min_cell_dimension}")
    print("---------------------------------------------")


    os.makedirs(output_dir, exist_ok=True)
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {image_path}, Shape: {original_image.shape}")
    img_height, img_width = original_image.shape[:2]

    hough_min_line_length = int(min(img_width, img_height) * 0.30) # 30% of the smaller dimension
    hough_min_line_length = max(hough_min_line_length, min_cell_dimension * 2) 
    print(f"  Dynamic Hough Min Line Length: {hough_min_line_length}")


    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, blur_kernel_size, 0)
    
    adaptive_th_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, adaptive_thresh_block_size, adaptive_thresh_C)
    cv2.imwrite(os.path.join(output_dir, "_debug_adaptive_thresh.png"), adaptive_th_image)


    edges = cv2.Canny(adaptive_th_image, canny_low_thresh, canny_high_thresh)
    cv2.imwrite(os.path.join(output_dir, "_debug_edges.png"), edges)


    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold,
                            minLineLength=hough_min_line_length,
                            maxLineGap=hough_max_line_gap)

    if lines is None:
        print("No line segments detected by HoughLinesP. Parameters might be too strict. Check debug images.")
        return

    horizontal_coords = []
    vertical_coords = []
    max_angle_dev_rad = np.deg2rad(max_angle_deviation_deg)
    
    debug_image_lines = original_image.copy() 

    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        cv2.line(debug_image_lines, (x1, y1), (x2, y2), (0, 0, 255), 1) 

        delta_x = x2 - x1
        delta_y = y2 - y1
        
        if delta_x == 0: 
            angle_rad = np.pi / 2
        else:
            angle_rad = np.arctan(abs(delta_y) / abs(delta_x))

        if angle_rad < max_angle_dev_rad:  
            horizontal_coords.append((y1 + y2) / 2)
        elif abs(angle_rad - np.pi/2) < max_angle_dev_rad : 
             vertical_coords.append((x1 + x2) / 2)
            
    cv2.imwrite(os.path.join(output_dir, "_debug_detected_lines_all.png"), debug_image_lines)

    print(f"Raw candidate horizontal line positions (Y): {len(horizontal_coords)}")
    print(f"Raw candidate vertical line positions (X): {len(vertical_coords)}")

    processed_horz_lines = _process_line_coords(horizontal_coords, min_cell_dimension, img_height, line_cluster_separation)
    processed_vert_lines = _process_line_coords(vertical_coords, min_cell_dimension, img_width, line_cluster_separation)

    print(f"Processed horizontal grid lines (Y): {processed_horz_lines}")
    print(f"Processed vertical grid lines (X): {processed_vert_lines}")

    debug_processed_lines_image = original_image.copy()
    for y_coord in processed_horz_lines:
        cv2.line(debug_processed_lines_image, (0, int(y_coord)), (img_width, int(y_coord)), (0,255,255), 1) 
    for x_coord in processed_vert_lines:
        cv2.line(debug_processed_lines_image, (int(x_coord), 0), (int(x_coord), img_height), (255,0,255), 1) 
    cv2.imwrite(os.path.join(output_dir, "_debug_processed_lines.png"), debug_processed_lines_image)


    cell_count = 0
    if len(processed_horz_lines) < 2 or len(processed_vert_lines) < 2:
        print("Not enough distinct grid lines found to form cells after processing. Check debug images, especially _debug_processed_lines.png.")
        return

    for i in range(len(processed_horz_lines) - 1):
        y_start = int(processed_horz_lines[i]) 
        y_end = int(processed_horz_lines[i+1])
        if y_end <= y_start: continue

        for j in range(len(processed_vert_lines) - 1):
            x_start = int(processed_vert_lines[j]) 
            x_end = int(processed_vert_lines[j+1])
            if x_end <= x_start: continue

            if (y_end - y_start >= min_cell_dimension) and \
               (x_end - x_start >= min_cell_dimension):
                cell = original_image[y_start:y_end, x_start:x_end]
                if cell.size == 0:
                    continue
                cell_filename = os.path.join(output_dir, f"cell_{i:02d}_{j:02d}.png")
                cv2.imwrite(cell_filename, cell)
                cell_count += 1

    print(f"Successfully cropped and saved {cell_count} cells to '{output_dir}'.")

if __name__ == "__main__":
    user_image_hedgehog = "ChatGPT Image May 28, 2025 at 03_27_19 AM.jpg"
    user_image_warrior = "ChatGPT Image May 28, 2025 at 03_33_53 AM.jpg" 
    
    dummy_image_name = "z_GPT_Templates/sample2.png" # Incremented dummy name
    image_to_process = None

    # --- Select which image to process ---
    # <<<< ENSURE THIS IS POINTING TO YOUR WARRIOR IMAGE >>>>
    image_to_process = user_image_warrior   
    # image_to_process = user_image_hedgehog 
    # -------------------------------------

    if image_to_process and not os.path.exists(image_to_process):
        print(f"ERROR: Selected image '{image_to_process}' not found. Please ensure it's in the correct path or in the same directory as the script.")
        image_to_process = None 

    if not image_to_process: 
        print(f"Primary choice for image processing not found or not set.")
        if os.path.exists(user_image_warrior): 
            image_to_process = user_image_warrior
            print(f"Falling back to default warrior image: {image_to_process}")
        elif os.path.exists(user_image_hedgehog):
            image_to_process = user_image_hedgehog
            print(f"Falling back to default hedgehog image: {image_to_process}")
        elif os.path.exists(dummy_image_name):
            image_to_process = dummy_image_name
            print(f"Falling back to existing dummy image: {image_to_process}")
        else:
            print(f"All specified images not found. Creating a new dummy image: '{dummy_image_name}' for testing.")
            img_h, img_w = 330, 430 
            dummy = np.ones((img_h, img_w, 3), dtype=np.uint8) * 230 
            line_color = (40, 40, 40) 
            line_thickness = 1 
            cv2.line(dummy, (img_w//3, 0), (img_w//3, img_h), line_color, line_thickness)
            cv2.line(dummy, (2*img_w//3, 0), (2*img_w//3, img_h), line_color, line_thickness)
            cv2.line(dummy, (0, img_h//3), (img_w, img_h//3), line_color, line_thickness)
            cv2.line(dummy, (0, 2*img_h//3), (img_w, 2*img_h//3), line_color, line_thickness)
            cv2.imwrite(dummy_image_name, dummy)
            image_to_process = dummy_image_name
    
    output_directory = "output_cells_tuned_v3" # New output directory
    
    if image_to_process and os.path.exists(image_to_process): 
        print(f"Processing image: {image_to_process}")
        print(f"Output will be saved to: {output_directory}")
        split_image_into_cells(image_to_process, output_directory)
    else:
        print(f"CRITICAL ERROR: No image to process. Image '{image_to_process if image_to_process else 'None'}' was not found.")

