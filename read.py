import os
import tempfile

import img2pdf
import pyocr
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
from pytesseract import Output
import numpy as np


TARGET_HEIGHT = 2048
RESIZE_FACTOR = 1


def resize_image_to_height(
    image: np.ndarray, target_height: int | None = TARGET_HEIGHT
):
    if target_height is None:
        return image

    resize_factor = target_height / image.shape[0]

    resized_image = cv2.resize(
        image, (int(image.shape[1] * resize_factor), target_height)
    )

    return resized_image, resize_factor


def get_image_data(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image, resize_factor = resize_image_to_height(image)

    # image = cv2.GaussianBlur(image, (9, 9), 0)
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # image = cv2.filter2D(image, -1, kernel)
    # image = cv2.threshold(image, 135, 255, cv2.THRESH_BINARY)[1]
    # image = cv2.inRange(image, (0, 0, 123), (179, 255, 255))
    # save_image
    cv2.imwrite("output.png", image)

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
                "left": data["left"][i] / resize_factor,
                "top": data["top"][i] / resize_factor,
                "width": data["width"][i] / resize_factor,
                "height": data["height"][i] / resize_factor,
            }
            coordinates.append(char_info)
    print(full_text)
    return full_text, coordinates


import easyocr
from PIL import Image

reader = easyocr.Reader(["ru"])


def get_image_data(image_path):
    # Загрузка изображения с использованием Pillow
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image.thumbnail(max_size, Image.LANCZOS)  # Уменьшение разрешения изображения
        # resized_size = image.size  # Сохраняем уменьшенные размеры изображения
    except Exception as e:
        raise ValueError(f"Unable to read image file: {e}")

    # Распознавание текста
    # smaller boxes
    result = reader.readtext(
        image,
        detail=1,
        width_ths=0.1,
        height_ths=0.1,
    )

    full_text = ""
    coordinates = []

    for bbox, text, prob in result:
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[2]

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

    print(full_text)

    return full_text, coordinates


def get_bounding_rectangles(i, j, full_text, coordinates):
    if i < 0 or j >= len(full_text) or i > j:
        raise ValueError(f"Invalid indices: {i=}, {j=}, {len(full_text)=}")

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
