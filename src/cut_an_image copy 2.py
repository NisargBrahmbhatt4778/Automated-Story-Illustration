import cv2
import numpy as np
import os
import math

def _process_line_coords(coords, min_cell_dimension, image_boundary_val, cluster_min_separation=10):
    """
    Processes a list of 1D coordinates (x or y) to find distinct grid lines.
    It sorts, clusters nearby coordinates, and ensures a minimum cell dimension.

    Args:
        coords (list): List of detected x or y coordinates.
        min_cell_dimension (int): Minimum desired width/height of a cell.
                                  Lines too close will be merged or ignored to meet this.
        image_boundary_val (int): The maximum boundary of the image (width or height).
        cluster_min_separation (int): Max distance for coordinates to be considered in the same cluster.

    Returns:
        list: Sorted list of unique grid line coordinates including 0 and image_boundary_val.
    """
    if not coords: # If no coords, return just the image boundaries
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
            if abs(coord - np.mean(current_cluster)) < cluster_min_separation:
                current_cluster.append(coord)
            else:
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
        if line_pos - final_grid_lines[-1] >= min_cell_dimension:
            final_grid_lines.append(line_pos)

    if not final_grid_lines or image_boundary_val - final_grid_lines[-1] >= min_cell_dimension:
        if image_boundary_val not in final_grid_lines:
             final_grid_lines.append(image_boundary_val)
    elif len(final_grid_lines) > 1 and image_boundary_val > final_grid_lines[-1] : 
        final_grid_lines[-1] = image_boundary_val
    elif not final_grid_lines and image_boundary_val > 0 : 
        final_grid_lines.append(image_boundary_val)
    
    return sorted(list(set(final_grid_lines)))


def split_image_into_cells(image_path, output_dir="output_cells_improved"):
    """
    Detects grid lines that span the image using adaptive thresholding and saves individual cells.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the cropped cells.
    """
    # --- Parameters (tweak these as needed) ---
    # Gaussian Blur
    blur_kernel_size = (5, 5) 
    # Adaptive Thresholding
    adaptive_thresh_block_size = 15 # Size of a pixel neighborhood that is used to calculate a threshold value
    adaptive_thresh_C = 7       # Constant subtracted from the mean or weighted mean
    # Canny Edge Detection
    canny_low_thresh = 50
    canny_high_thresh = 150
    # Hough Line Transform (HoughLinesP)
    hough_rho = 1  
    hough_theta = np.pi / 180  
    hough_threshold = 70        # Min votes. Increased slightly due to potentially more edges from adaptive_thresh
    # hough_min_line_length is now dynamic (see below)
    hough_max_line_gap = 30     # Max gap. Increased slightly.
    # Line classification & Spanning Check
    max_angle_deviation_deg = 5.0 
    spanning_line_edge_tolerance = 30 
    # Line processing (for _process_line_coords)
    line_cluster_separation = 25 
    min_cell_dimension = 20     

    # --- End Parameters ---

    os.makedirs(output_dir, exist_ok=True)
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {image_path}, Shape: {original_image.shape}")
    img_height, img_width = original_image.shape[:2]

    # Dynamically set hough_min_line_length
    hough_min_line_length = int(min(img_width, img_height) * 0.4) # 40% of the smaller dimension
    print(f"Dynamic hough_min_line_length: {hough_min_line_length}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, blur_kernel_size, 0)
    
    # Adaptive Thresholding
    # ADAPTIVE_THRESH_GAUSSIAN_C or ADAPTIVE_THRESH_MEAN_C
    # THRESH_BINARY_INV if grid lines are lighter than background/content
    # THRESH_BINARY if grid lines are darker
    # For the warrior image, grid lines seem lighter than the dark figures.
    # For the hedgehog, grid lines are darker than the light background.
    # We might need to inspect the images or make this a parameter.
    # Assuming grid lines are generally darker than their immediate surroundings OR we want to find dark lines on light bg
    # If grid lines are consistently lighter, use THRESH_BINARY_INV
    # Let's try THRESH_BINARY first, assuming we are looking for dark lines.
    # If your grid lines are white/light on a dark background, change to cv2.THRESH_BINARY_INV
    # For the warrior image, the grid lines are light gray on a darker cell background.
    # For the hedgehog, the grid lines are dark on a light background.
    # This is tricky. Let's assume we want to make the lines black for Canny.
    # If lines are lighter than content, inverting (THRESH_BINARY_INV) after adaptive thresholding might make them black.
    # If lines are darker than content, THRESH_BINARY is fine.
    
    # Let's try to make the grid lines stand out as white edges on black background for Canny
    # If grid lines are light (e.g. warrior image), we want them to become white after thresholding.
    # If grid lines are dark (e.g. hedgehog image), we want them to become white after thresholding.
    # This implies we might need to detect if lines are lighter or darker than their surroundings.
    # For simplicity, let's use THRESH_BINARY_INV, assuming it helps make grid lines prominent for Canny.
    # This means lighter regions become white, darker regions become black. If grid lines are light, they'll be white.
    # If grid lines are dark, they'll be black. Canny looks for gradients anyway.
    
    # A common strategy: make features of interest white, background black.
    # If grid lines are light on dark cells (warrior): THRESH_BINARY_INV might turn lines white, cells black.
    # If grid lines are dark on light cells (hedgehog): THRESH_BINARY might turn lines black, cells white.
    # Canny works on gradients, so the exact polarity might be less critical than clear separation.
    # Let's try Gaussian adaptive thresholding. It's often good for varying illumination.
    adaptive_th_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, adaptive_thresh_block_size, adaptive_thresh_C)
    # cv2.imwrite(os.path.join(output_dir, "_debug_adaptive_thresh.png"), adaptive_th_image)


    edges = cv2.Canny(adaptive_th_image, canny_low_thresh, canny_high_thresh)
    # cv2.imwrite(os.path.join(output_dir, "_debug_edges.png"), edges)


    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold,
                            minLineLength=hough_min_line_length,
                            maxLineGap=hough_max_line_gap)

    if lines is None:
        print("No line segments detected by HoughLinesP after adaptive thresholding. Try adjusting parameters.")
        return

    horizontal_coords = []
    vertical_coords = []
    max_angle_dev_rad = np.deg2rad(max_angle_deviation_deg)
    
    # debug_image_lines = original_image.copy()

    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        delta_x = x2 - x1
        delta_y = y2 - y1
        
        if delta_x == 0: 
            angle_rad = np.pi / 2
        else:
            angle_rad = np.arctan(abs(delta_y) / abs(delta_x))

        if angle_rad < max_angle_dev_rad:  
            line_min_x = min(x1, x2)
            line_max_x = max(x1, x2)
            if line_min_x < spanning_line_edge_tolerance and \
               line_max_x > img_width - spanning_line_edge_tolerance:
                horizontal_coords.append((y1 + y2) / 2)
                # cv2.line(debug_image_lines, (x1, y1), (x2, y2), (0, 255, 0), 1) 

        elif abs(angle_rad - np.pi/2) < max_angle_dev_rad : 
            line_min_y = min(y1, y2)
            line_max_y = max(y1, y2)
            if line_min_y < spanning_line_edge_tolerance and \
               line_max_y > img_height - spanning_line_edge_tolerance:
                vertical_coords.append((x1 + x2) / 2)
                # cv2.line(debug_image_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
    # cv2.imwrite(os.path.join(output_dir, "_debug_filtered_lines.png"), debug_image_lines)

    print(f"Raw candidate horizontal spanning line positions (Y): {len(horizontal_coords)}")
    print(f"Raw candidate vertical spanning line positions (X): {len(vertical_coords)}")

    processed_horz_lines = _process_line_coords(horizontal_coords, min_cell_dimension, img_height, line_cluster_separation)
    processed_vert_lines = _process_line_coords(vertical_coords, min_cell_dimension, img_width, line_cluster_separation)

    print(f"Processed horizontal grid lines (Y): {processed_horz_lines}")
    print(f"Processed vertical grid lines (X): {processed_vert_lines}")

    cell_count = 0
    if len(processed_horz_lines) < 2 or len(processed_vert_lines) < 2:
        print("Not enough distinct spanning grid lines found to form cells.")
        return

    for i in range(len(processed_horz_lines) - 1):
        y_start = processed_horz_lines[i]
        y_end = processed_horz_lines[i+1]
        if y_end <= y_start: continue

        for j in range(len(processed_vert_lines) - 1):
            x_start = processed_vert_lines[j]
            x_end = processed_vert_lines[j+1]
            if x_end <= x_start: continue

            if (y_end - y_start >= min_cell_dimension) and \
               (x_end - x_start >= min_cell_dimension):
                cell = original_image[y_start:y_end, x_start:x_end]
                if cell.size == 0:
                    print(f"Skipping empty cell at Y:({y_start}-{y_end}), X:({x_start}-{x_end})")
                    continue
                cell_filename = os.path.join(output_dir, f"cell_{i:02d}_{j:02d}.png")
                cv2.imwrite(cell_filename, cell)
                cell_count += 1
            else:
                 print(f"Skipping small cell (after final check) at Y:({y_start}-{y_end}), X:({x_start}-{x_end})")

    print(f"Successfully cropped and saved {cell_count} cells to '{output_dir}'.")

if __name__ == "__main__":
    user_image_hedgehog = "ChatGPT Image May 28, 2025 at 03_27_19 AM.jpg"
    user_image_warrior = "ChatGPT Image May 28, 2025 at 03_33_53 AM.jpg"
    
    dummy_image_name = "z_GPT_Templates/sample2.png"
    image_to_process = None

    # --- Select which image to process ---
    # image_to_process = user_image_hedgehog # For hedgehog
    image_to_process = user_image_warrior   # For warrior
    # -------------------------------------

    if image_to_process and not os.path.exists(image_to_process):
        print(f"Selected image '{image_to_process}' not found. Checking for other user images or dummy.")
        image_to_process = None # Reset

    if not image_to_process: # If primary choice not found or not set
        if os.path.exists(user_image_warrior): # Prioritize warrior if not explicitly chosen
            image_to_process = user_image_warrior
        elif os.path.exists(user_image_hedgehog):
            image_to_process = user_image_hedgehog
        elif os.path.exists(dummy_image_name):
            image_to_process = dummy_image_name
        else:
            print(f"User images and existing dummy not found. Creating a new dummy image: '{dummy_image_name}' for testing.")
            img_h, img_w = 330, 430 
            dummy = np.ones((img_h, img_w, 3), dtype=np.uint8) * 230 
            line_color = (40, 40, 40) 
            line_thickness = 2
            cv2.line(dummy, (img_w//3 + 5, 0), (img_w//3 + 5, img_h), line_color, line_thickness) # slightly offset
            cv2.line(dummy, (2*img_w//3 - 5, 0), (2*img_w//3 - 5, img_h), line_color, line_thickness)
            cv2.line(dummy, (0, img_h//2 + 5), (img_w, img_h//2 + 5), line_color, line_thickness)
            cv2.imwrite(dummy_image_name, dummy)
            image_to_process = dummy_image_name

    output_directory = "output_cells_improved" 
    
    if image_to_process:
        print(f"Processing image: {image_to_process}")
        print(f"Output will be saved to: {output_directory}")
        split_image_into_cells(image_to_process, output_directory)
    else:
        print("No image specified or found to process.")

