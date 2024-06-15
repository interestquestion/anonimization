import os
import tempfile

import img2pdf
import pytesseract
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image, ImageDraw
from pytesseract import Output
import numpy as np




def get_image_data(image_path):
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, lang="rus", output_type=Output.DICT)

    full_text = ""
    coordinates = []

    for i in range(len(data["level"])):
        if full_text and full_text[-1] != " ":
            full_text += " "
            coordinates.append(None)
        word = data["text"][i]
        for char in word:
            full_text += char
            char_info = {
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            }
            coordinates.append(char_info)
    return full_text, coordinates


# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor

# def get_image_data(image_path):
#     # Загрузка изображения с использованием Pillow
#     try:
#         image = Image.open(image_path)
#         image = image.convert("RGB")  # Убедимся, что изображение в RGB
#         image_np = np.array(image)
#     except Exception as e:
#         raise ValueError(f"Unable to read image file: {e}")

#     # Преобразование изображения в формат, поддерживаемый doctr

#     # Инициализация модели
#     model = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_mobilenet_v3_small')

#     image_np = np.expand_dims(image_np, axis=0)

#     # Распознавание текста
#     result = model(image_np)

#     full_text = ""
#     coordinates = []

#     # Проходим по результатам распознавания и извлекаем координаты каждого слова
#     pages = result.pages
#     for page in pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 for word in line.words:
#                     for char in word.value:
#                         full_text += char
#                         char_info = {
#                             "left": word.geometry[0][0] * page.dimensions[1],
#                             "top": word.geometry[0][1] * page.dimensions[0],
#                             "width": (word.geometry[1][0] - word.geometry[0][0]) * page.dimensions[1],
#                             "height": (word.geometry[1][1] - word.geometry[0][1]) * page.dimensions[0],
#                         }
#                         coordinates.append(char_info)
#                     # Добавляем пробел между словами
#                     full_text += " "
#                     coordinates.append(None)

#     # Удаляем последний лишний пробел и None
#     if full_text and full_text[-1] == " ":
#         full_text = full_text[:-1]
#         coordinates = coordinates[:-1]

#     return full_text, coordinates

import easyocr
from PIL import Image
reader = easyocr.Reader(['ru'])

def get_image_data(image_path):
    # Загрузка изображения с использованием Pillow
    try:
        image = Image.open(image_path)
        original_size = image.size  # Сохраняем оригинальные размеры изображения
        image = image.convert("RGB")  # Убедимся, что изображение в RGB
        # image.thumbnail(max_size, Image.LANCZOS)  # Уменьшение разрешения изображения
        # resized_size = image.size  # Сохраняем уменьшенные размеры изображения
    except Exception as e:
        raise ValueError(f"Unable to read image file: {e}")


    # Сохранение уменьшенного изображения во временный файл
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_file_path = temp_file.name

    # Распознавание текста
    result = reader.readtext(temp_file_path, detail=1)

    full_text = ""
    coordinates = []


    for (bbox, text, prob) in result:
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[2]
        
        # Преобразуем координаты обратно в оригинальный размер изображения
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        # Добавляем каждый символ в полный текст и сохраняем его координаты
        for char in text:
            full_text += char
            char_info = {
                "left": x_min,
                "top": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
            }
            coordinates.append(char_info)
        # Добавляем пробел между словами
        full_text += " "
        coordinates.append(None)

    # Удаляем последний лишний пробел и None
    if full_text and full_text[-1] == " ":
        full_text = full_text[:-1]
        coordinates = coordinates[:-1]

    return full_text, coordinates

def get_bounding_rectangles(i, j, full_text, coordinates):
    if i < 0 or j >= len(full_text) or i > j:
        raise ValueError("Invalid indices")

    selected_coords = [
        coord for k, coord in enumerate(coordinates[i : j + 1]) if coord is not None
    ]

    if not selected_coords:
        return []

    rectangles = []
    current_rect = selected_coords[0].copy()

    for coord in selected_coords[1:]:
        if (
            coord["top"] == current_rect["top"]
            and coord["height"] == current_rect["height"]
            and coord["left"] == current_rect["left"] + current_rect["width"]
        ):
            current_rect["width"] += coord["width"]
        else:
            rectangles.append(current_rect)
            current_rect = coord.copy()

    rectangles.append(current_rect)

    return rectangles


def draw_rectangles(image_path, output_path, rectangles):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for rect in rectangles:
        left = rect["left"]
        top = rect["top"]
        right = left + rect["width"]
        bottom = top + rect["height"]
        draw.rectangle([left, top, right, bottom], fill="white")

    image.save(output_path)


def pdf_to_images(pdf_path: str, output_path: str) -> None:
    with open(pdf_path, "rb") as f:
        images = convert_from_bytes(f.read())
    for i, image in enumerate(images):
        image.save(f"{output_path}/page_{i}.png", "PNG")


def images_to_pdf(images_path: str, output_path: str) -> None:
    def sort_key(img):
        try:
            return int(img.split("_")[-1].split(".")[0])
        except Exception:
            return img
    
    images = [
        f"{images_path}/{img}"
        for img in os.listdir(images_path)
        if img.endswith(".png")
    ]
    images.sort(key=sort_key)
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(images))
