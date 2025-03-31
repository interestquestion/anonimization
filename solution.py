from typing import Callable
from read import *
from parse import *
from read import logger  # Import the logger from read.py

# if windows
if os.name == "nt":
    PATH_TO_TESSARECT = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSARECT


def process_image(image_path: str, output_path: str, pd_funcs: list[Callable], get_image_data: Callable) -> None:
    # Call the image data function
    full_text, coordinates, was_rotated, actual_image_path = get_image_data(image_path)
    
    positions = []
    for pd_func in pd_funcs:
        try:
            positions.extend(pd_func(full_text))
        except Exception as e:
            print(f"Error in {pd_func.__name__}: {e}")
    
    rectangles = []
    for i, j in positions:
        rectangles.extend(get_bounding_rectangles(i, j, full_text, coordinates))

    # Use the rotated image path if image was rotated
    draw_rectangles(actual_image_path, output_path, rectangles)
    
    # If using a temporary rotated file, clean it up
    if was_rotated and actual_image_path != image_path:
        try:
            os.remove(actual_image_path)
            logger.info(f"Removed temporary rotated image: {actual_image_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary rotated image: {e}")
            
    return None

if __name__ == "__main__":
    image_path = "data_examples/Есть данные 3.jpg"
    output_path = "output1.png"
    pd_funcs = [get_all_names_mystem, find_numeric_sequences]
    process_image(image_path, output_path, pd_funcs)
