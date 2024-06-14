from read import *
PATH_TO_TESSARECT = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSARECT

image_path = "./data_examples/мивапрорп.png"
output_path = "./data_examples/output.png"

full_text, coordinates = get_image_data(image_path)

print(full_text)

print(coordinates[40:45])

i = 0 
j = len(full_text) - 1

rectangles = get_bounding_rectangles(i, j, full_text, coordinates)

draw_rectangles(image_path, output_path, rectangles)
