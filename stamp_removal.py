import cv2
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_round_stamps(image: np.ndarray, min_radius: int = 50, max_radius: int = 150, 
                        param1: int = 100, param2: int = 60, dp: float = 1.2,
                        color_filtering: bool = True, blue_threshold: float = 0.05,
                        dark_threshold: int = 150, bottom_half_only: bool = True) -> list:
    """
    Detect round stamps in an image using Hough Circle Transform.
    
    Args:
        image: Input image as numpy array
        min_radius: Minimum radius of circles to detect
        max_radius: Maximum radius of circles to detect
        param1: First parameter of Hough Circle Transform (edge detection sensitivity)
        param2: Second parameter of Hough Circle Transform (accumulator threshold, higher = fewer circles)
        dp: Inverse ratio of the accumulator resolution to the image resolution
        color_filtering: Whether to apply color filtering to detect only colored stamps
        blue_threshold: Minimum ratio of blue pixels to consider a circle as a stamp
        dark_threshold: Maximum average intensity to consider a circle as a dark stamp
        bottom_half_only: Whether to look for stamps only in the bottom half of the image
        
    Returns:
        List of detected circles as (x, y, radius)
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create a region of interest if bottom_half_only is enabled
    if bottom_half_only:
        mask = np.zeros_like(gray)
        height, width = gray.shape
        # Only look at the bottom third of the image for stamps
        mask[int(height * 0.65):, :] = 255
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_radius * 2,  # Minimum distance between detected centers
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Return empty list if no circles detected
    if circles is None:
        logger.info("No round stamps detected")
        return []
    
    # Convert to integer coordinates
    circles = np.round(circles[0, :]).astype(int)
    
    # Filter circles based on additional criteria
    filtered_circles = []
    
    for x, y, r in circles:
        # Skip circles that go outside the image
        if (x - r < 0 or x + r >= image.shape[1] or 
            y - r < 0 or y + r >= image.shape[0]):
            continue
        
        # Apply color filtering to detect only colored stamps (typically blue)
        if color_filtering and len(image.shape) == 3:
            # Extract the circular region
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Get the ROI
            roi = cv2.bitwise_and(image, image, mask=mask)
            
            # Count non-zero (stamp) pixels
            non_zero_pixels = np.count_nonzero(mask)
            if non_zero_pixels == 0:
                continue
                
            # Check for blue component (typical for stamps)
            # Split into color channels
            b, g, r_channel = cv2.split(roi)
            
            # For blue stamps:
            # Check for blue-dominant pixels (higher blue than red or green by a margin)
            blue_dominant = np.logical_and(
                b > r_channel + 20,  # More strict blue check
                b > g + 20
            )
            blue_pixels = np.count_nonzero(blue_dominant)
            
            # Calculate ratio of blue pixels to total non-zero pixels in the circle
            blue_ratio = blue_pixels / non_zero_pixels
            
            # Get average color of the stamp area
            avg_b = cv2.mean(b, mask=mask)[0]
            avg_g = cv2.mean(g, mask=mask)[0]
            avg_r = cv2.mean(r_channel, mask=mask)[0]
            
            # Get average intensity of the stamp area
            avg_intensity = cv2.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), mask=mask)[0]
            
            # If it has enough blue dominant pixels and is blue on average
            if blue_ratio >= blue_threshold and avg_b > avg_r and avg_b > avg_g:
                filtered_circles.append((x, y, r))
                continue
                
            # If it's dark enough but not too dark (could be a black stamp)
            if avg_intensity <= dark_threshold and avg_intensity > 50:
                # Check if it has a relatively uniform appearance (stamps usually do)
                std_dev = cv2.meanStdDev(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), mask=mask)[1][0][0]
                if std_dev < 40:  # More strict uniformity check
                    filtered_circles.append((x, y, r))
        else:
            # If no color filtering, just add the circle
            filtered_circles.append((x, y, r))
    
    logger.info(f"Detected {len(circles)} potential round stamps, filtered to {len(filtered_circles)} validated stamps")
    return filtered_circles

def remove_stamps(image: np.ndarray, circles: list, expansion_factor: float = 1.2, 
                  white_fill: bool = True, debug: bool = False) -> np.ndarray:
    """
    Remove detected circular stamps from an image.
    
    Args:
        image: Input image as numpy array
        circles: List of detected circles as (x, y, radius)
        expansion_factor: Factor to expand the circle radius for removal
        white_fill: Whether to replace stamps with white (True) or use inpainting (False)
        debug: Whether to save debug masks
        
    Returns:
        Image with stamps removed
    """
    # If no stamps detected, return the original image
    if not circles:
        return image.copy()
        
    # Create a mask for stamps
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw filled white circles on the mask where stamps are detected
    for x, y, r in circles:
        # Expand the radius slightly to ensure complete removal
        expanded_r = int(r * expansion_factor)
        cv2.circle(mask, (x, y), expanded_r, 255, -1)
    
    # Save mask for debugging if requested
    if debug:
        cv2.imwrite("debug_stamp_mask.png", mask)
        
        # Save a visualization of the detected stamps
        debug_image = image.copy()
        for x, y, r in circles:
            cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)
        cv2.imwrite("debug_stamp_detection.jpg", debug_image)
    
    if white_fill:
        # Create a white fill version
        result = image.copy()
        # Create a white background (255 for all channels)
        if len(image.shape) == 3:  # Color image
            white = np.ones_like(image) * 255
        else:  # Grayscale image
            white = np.ones_like(image) * 255
        
        # Replace masked areas with white
        result = np.where(mask[:, :, np.newaxis] == 255, white, result) if len(image.shape) == 3 else np.where(mask == 255, white, result)
    else:
        # Apply inpainting to remove the stamps
        result = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
    
    return result

def process_image_remove_stamps(image_path: str, output_path: str = None, 
                               min_radius: int = 50, max_radius: int = 150,
                               param1: int = 100, param2: int = 60, 
                               dp: float = 1.2, color_filtering: bool = True,
                               blue_threshold: float = 0.05,
                               dark_threshold: int = 150,
                               bottom_half_only: bool = True,
                               white_fill: bool = True,
                               debug: bool = False) -> np.ndarray:
    """
    Process an image to detect and remove round stamps.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (if None, image is not saved)
        min_radius: Minimum radius of stamps to detect
        max_radius: Maximum radius of stamps to detect
        param1: First parameter of Hough Circle Transform (edge detection sensitivity)
        param2: Second parameter of Hough Circle Transform (accumulator threshold, higher = fewer circles)
        dp: Inverse ratio of accumulator resolution
        color_filtering: Whether to apply color filtering to detect only colored stamps
        blue_threshold: Minimum ratio of blue pixels to consider a circle as a stamp
        dark_threshold: Maximum average intensity to consider a circle as a dark stamp
        bottom_half_only: Whether to look for stamps only in the bottom half of the image
        white_fill: Whether to replace stamps with white (True) or use inpainting (False)
        debug: Whether to save debug images
        
    Returns:
        Processed image with stamps removed
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create a directory for stamp removal results if it doesn't exist
    if output_path and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Detect round stamps
    circles = detect_round_stamps(
        image, 
        min_radius=min_radius,
        max_radius=max_radius,
        param1=param1,
        param2=param2,
        dp=dp,
        color_filtering=color_filtering,
        blue_threshold=blue_threshold,
        dark_threshold=dark_threshold,
        bottom_half_only=bottom_half_only
    )
    
    # If no stamps detected, return the original image
    if len(circles) == 0:
        logger.info("No stamps to remove")
        if output_path:
            cv2.imwrite(output_path, image)
        return image
    
    # Draw circles on a debug image to visualize detection
    if debug:
        debug_image = image.copy()
        for x, y, r in circles:
            cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)
        
        debug_out_path = output_path.replace(".jpg", "_detected.jpg") if output_path else "stamps_detected.jpg"
        cv2.imwrite(debug_out_path, debug_image)
        logger.info(f"Saved detection visualization to {debug_out_path}")
    
    # Remove the stamps
    result = remove_stamps(image, circles, white_fill=white_fill, debug=debug)
    
    # Save the result if output path is provided
    if output_path:
        cv2.imwrite(output_path, result)
        logger.info(f"Saved stamp-removed image to {output_path}")
    
    return result

if __name__ == "__main__":
    # Test the stamp removal on a sample image
    test_image = "data_examples/photo_2024-04-08_13-36-14 (2).jpg"
    output_image = "stamp_removal_results/stamp_removed.jpg"
    
    process_image_remove_stamps(
        test_image,
        output_image,
        min_radius=50,
        max_radius=150,
        param1=100,
        param2=60,
        color_filtering=True,
        blue_threshold=0.05,
        dark_threshold=150,
        bottom_half_only=True,
        white_fill=True,
        debug=True
    ) 