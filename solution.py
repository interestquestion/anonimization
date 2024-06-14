from typing import Callable
from read import *
from parse import *


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
    image_path = "data_examples/мивапрорп.png"
    output_path = "output1.png"
    pd_funcs = [get_all_names_mystem, get_all_addresses_natasha]
    process_image(image_path, output_path, pd_funcs)
