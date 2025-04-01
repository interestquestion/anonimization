from typing import Callable
from read import *
from parse import *
from read import logger  # Import the logger from read.py
from stamp_removal import process_image_remove_stamps, process_image_remove_stamps_contour_based, find_blue_squares
import tempfile

# if windows
if os.name == "nt":
    PATH_TO_TESSARECT = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSARECT


def process_image(image_path: str, output_path: str, pd_funcs: list[Callable], get_image_data: Callable, 
                  stamp_removal_method: str = "contour", stamp_params: dict = None) -> None:
    """
    Process an image to anonymize specified personal data.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image
        pd_funcs: List of functions to detect personal data
        get_image_data: Function to extract text and coordinates from image
        remove_stamps: Whether to remove round stamps from the image
        blue_remove_stamps_and_signs: Whether to remove blue squares (typically signatures)
        stamp_params: Parameters for stamp removal (optional)
        stamp_removal_method: Method to use for stamp removal ('contour' or 'circle')
    """
    # Apply stamp removal if requested
    actual_image_path = image_path
    temp_path = None
    temp_dir = None
    
    if stamp_removal_method in ["circle", "contour"]:
        logger.info(f"Removing stamps from the image using {stamp_removal_method} method...")
        
        # Create a temporary directory for the stamp-removed image
        temp_dir = tempfile.TemporaryDirectory()
        base_name = os.path.basename(image_path)
        temp_path = os.path.join(temp_dir.name, f"nostamp_{base_name}")
        
        if stamp_removal_method == "circle":
            # Default parameters for circle-based stamp removal
            default_params = {
                "min_radius": 50,
                "max_radius": 150,
                "param1": 100,
                "param2": 60,
                "dp": 1.2,
                "color_filtering": True,
                "blue_threshold": 0.05,
                "dark_threshold": 150,
                "bottom_half_only": False,
                "white_fill": True,
                "debug": False
            }
            
            # Use provided parameters or defaults
            params = {**default_params, **(stamp_params or {})}
            
            # Process the image to remove stamps using circle detection
            process_image_remove_stamps(
                image_path,
                temp_path,
                **params
            )
        else:  # default to contour method
            # Default parameters for contour-based stamp removal
            default_params = {
                "min_area": 500,
                "ratio_threshold": 0.65,
                "min_radius_ratio": 0.03,
                "debug": False
            }
            
            # Use provided parameters or defaults
            params = {**default_params, **(stamp_params or {})}
            
            # Process the image to remove stamps using contour-based method
            process_image_remove_stamps_contour_based(
                image_path,
                temp_path,
                **params
            )
        
        # Use the stamp-removed image for further processing
        actual_image_path = temp_path
    
    # Call the image data function
    full_text, coordinates, was_rotated, ocr_image_path = get_image_data(actual_image_path)
    
    positions = []
    for pd_func in pd_funcs:
        try:
            positions.extend(pd_func(full_text))
        except Exception as e:
            print(f"Error in {pd_func.__name__}: {e}")
    
    rectangles = []
    for i, j in positions:
        rectangles.extend(get_bounding_rectangles(i, j, full_text, coordinates))
    
    if stamp_removal_method == "blue":

        rectangles += find_blue_squares(actual_image_path)

    # Use the rotated image path if image was rotated
    draw_rectangles(ocr_image_path, output_path, rectangles)
    
    # Clean up temporary files
    if was_rotated and ocr_image_path != image_path and ocr_image_path != temp_path:
        try:
            os.remove(ocr_image_path)
            logger.info(f"Removed temporary rotated image: {ocr_image_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary rotated image: {e}")
    
    # Clean up the temporary directory when done
    if temp_dir:
        temp_dir.cleanup()
        logger.info("Cleaned up temporary directory for stamp removal")
            
    return None

if __name__ == "__main__":
    image_path = "data_examples/photo_2024-04-08_13-36-14 (2).jpg"
    output_path = "output1.png"
    pd_funcs = [get_all_names_mystem, find_numeric_sequences]
    
    # Example of using the stamp removal feature
    process_image(
        image_path, 
        output_path, 
        pd_funcs, 
        get_image_data_tesseract,
        remove_stamps=True,
        stamp_params={
            "debug": True
        }
    )
