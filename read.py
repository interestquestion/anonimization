import os

import img2pdf
import pytesseract
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image, ImageDraw
from pytesseract import Output


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
