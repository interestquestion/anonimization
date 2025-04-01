import os
import subprocess
import time
import logging
import cv2
import img2pdf
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
from pytesseract import Output

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LibreOffice service
try:
    soffice_process = subprocess.Popen(
        ['/usr/bin/libreoffice', '--headless', '--accept=socket,host=127.0.0.1,port=2002;urp;', 
         '--norestore', '--nodefault', '--nofirststartwizard'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Give it some time to start
    time.sleep(1)
    logger.info("LibreOffice service started.")
except Exception as e:
    soffice_process = None

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


def detect_and_correct_orientation(image):
    """
    Detects the orientation of text in an image using Tesseract OSD (Orientation and Script Detection)
    and rotates the image to correct the orientation.
    
    Args:
        image: The input image as numpy array
        
    Returns:
        The rotated image if orientation correction was needed, otherwise the original image
    """
    try:
        # Using tesseract OSD to detect orientation
        osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
        angle = osd['rotate']
        
        # If angle is 0, no rotation needed
        if angle == 0:
            return image, False
        
        angle = 360 - angle

        if angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        return rotated, True
    except Exception as e:
        # If orientation detection fails, return original image
        logger.warning(f"Orientation detection failed: {e}")
        return image, False


def rotate_and_save_image(image_path, auto_rotate=False):
    """
    Reads an image, potentially rotates it based on text orientation,
    and saves it to a temporary file if rotated.
    
    Args:
        image_path: Path to the input image
        auto_rotate: Whether to perform auto-rotation
        
    Returns:
        Path to the rotated image (or original if no rotation was needed or requested)
        and whether the image was rotated
    """
    if not auto_rotate:
        return image_path, False
    
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return image_path, False
            
        # Convert to grayscale for processing
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect and correct orientation
        rotated_image, was_rotated = detect_and_correct_orientation(gray_image)
        
        if not was_rotated:
            return image_path, False
            
        # Create a temporary file for the rotated image
        image_ext = os.path.splitext(image_path)[1]
        temp_rotated_path = os.path.join(
            os.path.dirname(image_path),
            f"rotated_{os.path.basename(image_path)}"
        )
        
        # Convert grayscale back to BGR for saving if it was rotated
        if len(rotated_image.shape) == 2:
            rotated_image_bgr = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)
        else:
            rotated_image_bgr = rotated_image
            
        # Save the rotated image
        cv2.imwrite(temp_rotated_path, rotated_image_bgr)
        logger.info(f"Saved rotated image to {temp_rotated_path}")
        
        return temp_rotated_path, True
        
    except Exception as e:
        logger.error(f"Error in rotate_and_save_image: {e}")
        return image_path, False


def get_image_data_tesseract(image_path, auto_rotate=False):
    """Get text and coordinates from an image using Tesseract OCR with optional rotation."""
    # First handle rotation if needed
    was_rotated = False
    actual_image_path = image_path
    
    if auto_rotate:
        actual_image_path, was_rotated = rotate_and_save_image(image_path, auto_rotate)
    
    image = cv2.imread(actual_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image, resize_factor = resize_image_to_height(image)

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
    
    # Always return four values for consistency
    return full_text, coordinates, was_rotated, actual_image_path


def get_bounding_rectangles(i, j, full_text, coordinates):
    if i < 0 or j >= len(full_text) or i > j:
        return []

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


def docx_to_pdf(docx_path: str, pdf_output_path: str) -> None:
    """Convert a .doc or .docx file to PDF using unoconv (LibreOffice)
    
    Args:
        docx_path: Path to the input .doc or .docx file
        pdf_output_path: Path to save the output PDF
    """
    try:
        if os.name == "nt":  # Windows
            # Use the full path to unoconv in LibreOffice installation
            unoconv_path = os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 
                                      'LibreOffice', 'program', 'python.exe')
            subprocess.run([unoconv_path, 'unoconv', '-f', 'pdf', '-o', pdf_output_path, docx_path], 
                          check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:  # Linux/Mac
            subprocess.run(['unoconv', '-f', 'pdf', '-o', pdf_output_path, docx_path], 
                          check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Successfully converted document to PDF")
    except Exception as e:
        error_msg = f"Failed to convert document to PDF: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
