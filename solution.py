from typing import Callable
from read import *
from parse import *

# if windows
if os.name == "nt":
    PATH_TO_TESSARECT = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSARECT


def process_image(image_path: str, output_path: str, pd_funcs: list[Callable], get_image_data: Callable) -> None:
    full_text, coordinates = get_image_data(image_path)
    positions = []
    for pd_func in pd_funcs:
        try:
            positions.extend(pd_func(full_text))
        except Exception as e:
            print(f"Error in {pd_func.__name__}: {e}")
    
    rectangles = []
    for i, j in positions:
        rectangles.extend(get_bounding_rectangles(i, j, full_text, coordinates))

    draw_rectangles(image_path, output_path, rectangles)
    return None

if __name__ == "__main__":
    image_path = "data_examples/Есть данные 3.jpg"
    output_path = "output1.png"
    pd_funcs = [get_all_names_mystem, find_numeric_sequences]
    process_image(image_path, output_path, pd_funcs)
