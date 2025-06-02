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

    # Sort and unique-fy initial coords that are within image bounds
    # Ensure coords are integers for consistent processing
    valid_coords = [int(round(c)) for c in coords if 0 <= c <= image_boundary_val]
    if not valid_coords:
        return sorted(list(set([0, image_boundary_val])))
        
    sorted_coords = sorted(list(set(valid_coords)))


    # Cluster nearby coordinates
    clustered_lines = []
    if sorted_coords: # Proceed only if there are valid sorted coordinates
        current_cluster = [sorted_coords[0]]
        for i in range(1, len(sorted_coords)):
            coord = sorted_coords[i]
            if abs(coord - np.mean(current_cluster)) < cluster_min_separation:
                current_cluster.append(coord)
            else:
                clustered_lines.append(int(round(np.mean(current_cluster))))
                current_cluster = [coord]
        if current_cluster: # Add the last cluster
            clustered_lines.append(int(round(np.mean(current_cluster))))

    # Ensure minimum cell dimension and include boundaries
    final_grid_lines = [0]
    # Use unique, sorted cluster means
    unique_clustered_lines = sorted(list(set(clustered_lines)))

    for line_pos in unique_clustered_lines:
        if line_pos <= 0: # Skip lines at or before the start boundary if already added (0)
            continue
        if line_pos >= image_boundary_val: # Skip lines at or after the end boundary
            continue
        if line_pos - final_grid_lines[-1] >= min_cell_dimension:
            final_grid_lines.append(line_pos)
        # If line_pos is very close to the last added line but further than it,
        # we could update final_grid_lines[-1]. For simplicity, prioritize min_cell_dimension.

    # Ensure the image boundary is the last line
    # Only add if it's significantly different from the last detected line or if no lines were detected
    if not final_grid_lines or image_boundary_val - final_grid_lines[-1] >= min_cell_dimension:
        if image_boundary_val not in final_grid_lines: # Avoid duplicates if already added
             final_grid_lines.append(image_boundary_val)
    elif len(final_grid_lines) > 1 and image_boundary_val > final_grid_lines[-1] : 
        # Boundary is too close to the last line, but greater. Extend last cell to boundary.
        final_grid_lines[-1] = image_boundary_val
    elif not final_grid_lines and image_boundary_val > 0 : # Only [0] is in list, and image_boundary is >0
        final_grid_lines.append(image_boundary_val)
    
    return sorted(list(set(final_grid_lines))) # Ensure uniqueness and sort


def split_image_into_cells(image_path, output_dir="output_cells"):
    """
    Detects grid lines that span the image and saves the individual cells.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the cropped cells.
    """
    # --- Parameters (tweak these as needed) ---
    # Canny Edge Detection
    canny_low_thresh = 50
    canny_high_thresh = 150
    # Gaussian Blur
    blur_kernel_size = (5, 5)
    # Hough Line Transform (HoughLinesP)
    hough_rho = 1  # distance resolution in pixels of the Hough grid
    hough_theta = np.pi / 180  # angular resolution in radians
    hough_threshold = 50  # min votes (intersections). Lowered for potentially weaker full lines.
    hough_min_line_length = 50 # min pixels for a line. Can be a fraction of image dim.
    hough_max_line_gap = 25  # max gap between connectable line segments. Increased slightly.
    # Line classification & Spanning Check
    max_angle_deviation_deg = 5.0 # Max deviation from pure horizontal/vertical
    spanning_line_edge_tolerance = 30 # Pixels: how close to image edge a line must reach
    # Line processing (for _process_line_coords)
    line_cluster_separation = 20 # Max distance for raw detected lines to be considered part of the same "thick" grid line
    min_cell_dimension = 20     # Minimum width/height for a cell to be saved

    # --- End Parameters ---

    os.makedirs(output_dir, exist_ok=True)
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {image_path}, Shape: {original_image.shape}")
    img_height, img_width = original_image.shape[:2]

    # Adjust min_line_length to be a fraction of the smaller image dimension
    # This helps ensure detected lines are significant relative to image size.
    # You can also keep it as a fixed value if preferred.
    # hough_min_line_length = min(img_width, img_height) * 0.5 # e.g., 50% of smaller dimension

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, blur_kernel_size, 0)
    edges = cv2.Canny(blurred_image, canny_low_thresh, canny_high_thresh)
    
    # For debugging, save the edge-detected image
    # cv2.imwrite(os.path.join(output_dir, "_debug_edges.png"), edges)

    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold,
                            minLineLength=hough_min_line_length,
                            maxLineGap=hough_max_line_gap)

    if lines is None:
        print("No line segments detected by HoughLinesP. Try adjusting parameters.")
        return

    horizontal_coords = []
    vertical_coords = []
    max_angle_dev_rad = np.deg2rad(max_angle_deviation_deg)
    
    # Create a copy of the original image to draw lines for debugging
    # debug_image_lines = original_image.copy()

    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]

        # Calculate angle of the line segment
        delta_x = x2 - x1
        delta_y = y2 - y1
        
        if delta_x == 0: # Vertical line
            angle_rad = np.pi / 2
        else:
            angle_rad = np.arctan(abs(delta_y) / abs(delta_x))

        # Check for horizontal lines spanning the image width
        if angle_rad < max_angle_dev_rad:  # Potential horizontal line
            # Check if it spans horizontally
            line_min_x = min(x1, x2)
            line_max_x = max(x1, x2)
            if line_min_x < spanning_line_edge_tolerance and \
               line_max_x > img_width - spanning_line_edge_tolerance:
                horizontal_coords.append((y1 + y2) / 2)
                # cv2.line(debug_image_lines, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for accepted horizontal

        # Check for vertical lines spanning the image height
        elif abs(angle_rad - np.pi/2) < max_angle_dev_rad : # Potential vertical line
            # Check if it spans vertically
            line_min_y = min(y1, y2)
            line_max_y = max(y1, y2)
            if line_min_y < spanning_line_edge_tolerance and \
               line_max_y > img_height - spanning_line_edge_tolerance:
                vertical_coords.append((x1 + x2) / 2)
                # cv2.line(debug_image_lines, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for accepted vertical
        # else:
            # cv2.line(debug_image_lines, (x1, y1), (x2, y2), (0, 0, 255), 1) # Red for rejected lines
            
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
        # Optionally save the whole image if no grid is robustly found
        # cv2.imwrite(os.path.join(output_dir, "full_image_no_grid.png"), original_image)
        return

    for i in range(len(processed_horz_lines) - 1):
        y_start = processed_horz_lines[i]
        y_end = processed_horz_lines[i+1]

        if y_end <= y_start: continue # Should not happen with sorted unique lines from _process_line_coords

        for j in range(len(processed_vert_lines) - 1):
            x_start = processed_vert_lines[j]
            x_end = processed_vert_lines[j+1]

            if x_end <= x_start: continue # Should not happen

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
    # --- IMPORTANT: Replace with the actual path to YOUR image ---
    # Default to a dummy image if specific user images are not found or provided.
    
    # Define user image paths (these should be in the same directory as the script or use full paths)
    user_image_1 = "ChatGPT Image May 28, 2025 at 03_27_19 AM.jpg"
    user_image_2 = "ChatGPT Image May 28, 2025 at 03_33_53 AM.jpg"
    
    # Define a simple dummy image name for creation if needed
    dummy_image_name = "z_GPT_Templates/sample2.png"
    image_to_process = None

    if os.path.exists(user_image_1):
        image_to_process = user_image_1
        print(f"Using user image: {image_to_process}")
    elif os.path.exists(user_image_2):
        image_to_process = user_image_2
        print(f"Using user image: {image_to_process}")
    elif os.path.exists(dummy_image_name):
        image_to_process = dummy_image_name
        print(f"User images not found. Using existing dummy image: {dummy_image_name}")
    else:
        print(f"User images and existing dummy not found. Creating a new dummy image: '{dummy_image_name}' for testing.")
        # Create a simple 3 rows x 2 columns grid image
        img_h, img_w = 320, 420 # Slightly different dimensions
        dummy = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240 # Light gray background
        line_color = (50, 50, 50) # Dark gray lines
        line_thickness = 2
        
        # Vertical lines (spanning full height)
        cv2.line(dummy, (img_w//3, 0), (img_w//3, img_h), line_color, line_thickness)
        cv2.line(dummy, (2*img_w//3, 0), (2*img_w//3, img_h), line_color, line_thickness)
        
        # Horizontal lines (spanning full width)
        cv2.line(dummy, (0, img_h//2), (img_w, img_h//2), line_color, line_thickness)
        # cv2.line(dummy, (0, 2*img_h//3), (img_w, 2*img_h//3), line_color, line_thickness) # For 3 rows

        cv2.imwrite(dummy_image_name, dummy)
        image_to_process = dummy_image_name
        print(f"Using newly created dummy image: {image_to_process}")

    output_directory = "output_cells_spanning" # Changed output dir name slightly
    
    if image_to_process:
        print(f"Processing image: {image_to_process}")
        print(f"Output will be saved to: {output_directory}")
        split_image_into_cells(image_to_process, output_directory)
    else:
        print("No image specified or found to process.")

