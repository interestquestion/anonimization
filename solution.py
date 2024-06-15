from typing import Callable
from read import *
from parse import *

PATH_TO_TESSARECT = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSARECT


def process_image(image_path: str, output_path: str, pd_funcs: list[Callable]) -> None:
    full_text, coordinates = get_image_data(image_path)
    positions = []
    for pd_func in pd_funcs:
        positions.extend(pd_func(full_text))
    
    rectangles = []
    for i, j in positions:
        rectangles.extend(get_bounding_rectangles(i, j, full_text, coordinates))

    draw_rectangles(image_path, output_path, rectangles)
    return None

if __name__ == "__main__":
    image_path = "data_examples/Есть данные 3.jpg"
    output_path = "output1.png"
    pd_funcs = [get_all_names_mystem, get_all_addresses_natasha, get_bd_positions, ]
    process_image(image_path, output_path, pd_funcs)
