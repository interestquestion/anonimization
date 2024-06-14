from read import *
# PATH_TO_TESSARECT = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSARECT

image_path = "./data_examples/мивапрорп.png"
image_path = "data_examples/20211119_150847.jpg.2a7914890644c5a31427fd321e1dac2e.jpg"
# image_path = "data_examples/pushkin.jpg"
output_path = "./data_examples/output.png"

full_text, coordinates = get_image_data(image_path)

print(full_text)

# print(coordinates[40:45])

i = 0 
j = len(full_text) - 1

# word = "знакомого"
# word_index = full_text.lower().find(word)
# word_end = word_index + len(word) - 1
# i, j = word_index, word_end

rectangles = get_bounding_rectangles(i, j, full_text, coordinates)

draw_rectangles(image_path, output_path, rectangles)
