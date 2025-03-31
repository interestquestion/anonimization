import os
import tempfile
from typing import Literal

from fastapi import FastAPI, File, Response, UploadFile

from parse import *
from read import (
    get_image_data_easyocr,
    get_image_data_tesseract,
    images_to_pdf,
    pdf_to_images,
    docx_to_pdf,
)
from solution import process_image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


app = FastAPI()


@app.get("/")
async def read_root():
    return "Сервис для анонимизации документов. Загрузите изображение с текстом и получите его с анонимизированными персональными данными."


# upload image and return modified file
@app.post(
    "/upload/",
    responses={
        200: {"content": {"image/png": {}, "application/pdf": {}}},
        415: {"description": "Unsupported file format. Please upload a .pdf, .png, .jpg, .jpeg, .doc, or .docx file."},
    },
    response_class=Response,
)
async def upload(
    file: UploadFile = File(...),
    ocr_engine: Literal["tesseract", "easyocr"] = "tesseract",
    date_year_max: int = 2016,
    auto_rotate: bool = True,
    remove_stamps: bool = False,
) -> Response:
    pd_funcs = [
        get_all_names_mystem,
        extract_complex_address_indices,
        get_phone_numbers_positions,
        lambda text: get_bd_positions(text, date_year_max),
        get_specific_numbers,
        find_numeric_sequences,
    ]
    
    # Auto-rotate is only supported with Tesseract engine
    if auto_rotate and ocr_engine != "tesseract":
        auto_rotate = False
        logger.warning("Auto-rotate is only supported with Tesseract engine. Disabling auto-rotate.")
    
    # Choose the appropriate OCR function based on engine
    get_image_data = get_image_data_tesseract if ocr_engine == "tesseract" else get_image_data_easyocr
    
    # Only providing the remove_stamps parameter, no detailed stamp parameters in API
    stamp_params = None

    # Create a temporary directory for all files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            input_filename = file.filename
            input_filename_ext = input_filename.split(".")[-1]
            
            # Create temporary files in the temporary directory
            tmp_input_path = os.path.join(temp_dir, f"input.{input_filename_ext}")
            with open(tmp_input_path, "wb") as tmp_input_file:
                tmp_input_file.write(await file.read())
            
            output_path = os.path.join(temp_dir, f"output.{input_filename_ext}")

            if input_filename.endswith(".pdf"):
                images_dir = os.path.join(temp_dir, "pdf_images")
                output_images_dir = os.path.join(temp_dir, "output_images")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(output_images_dir, exist_ok=True)
                
                pdf_to_images(tmp_input_path, images_dir)
                for img in os.listdir(images_dir):
                    process_image(
                        f"{images_dir}/{img}",
                        f"{output_images_dir}/{img}",
                        pd_funcs,
                        lambda img_path: get_image_data(img_path, auto_rotate),
                        remove_stamps=remove_stamps,
                        stamp_params=stamp_params,
                    )
                images_to_pdf(output_images_dir, output_path)
                
            elif input_filename.endswith((".doc", ".docx")):
                # Create temporary directories for conversion
                tmp_pdf_path = os.path.join(temp_dir, "temp.pdf")
                images_dir = os.path.join(temp_dir, "doc_images")
                output_images_dir = os.path.join(temp_dir, "output_images")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(output_images_dir, exist_ok=True)
                
                # Convert doc/docx to PDF
                docx_to_pdf(tmp_input_path, tmp_pdf_path)
                
                # Convert PDF to images
                pdf_to_images(tmp_pdf_path, images_dir)
                
                # Process each image
                for img in os.listdir(images_dir):
                    process_image(
                        f"{images_dir}/{img}",
                        f"{output_images_dir}/{img}",
                        pd_funcs,
                        lambda img_path: get_image_data(img_path, auto_rotate),
                        remove_stamps=remove_stamps,
                        stamp_params=stamp_params,
                    )
                
                # Convert processed images back to PDF
                images_to_pdf(output_images_dir, output_path)
                
            elif input_filename.endswith((".png", ".jpg", ".jpeg")):
                process_image(
                    tmp_input_path,
                    output_path,
                    pd_funcs,
                    lambda img_path: get_image_data(img_path, auto_rotate),
                    remove_stamps=remove_stamps,
                    stamp_params=stamp_params,
                )
            else:
                return Response(
                    content="Unsupported file format. Please upload a .pdf, .png, .jpg, .jpeg, .doc, or .docx file.",
                    status_code=415,
                )

            # Read the output file
            with open(output_path, "rb") as f:
                output_content = f.read()

            return Response(
                content=output_content,
                media_type=(
                    "image/png"
                    if input_filename.endswith((".png", ".jpg", ".jpeg"))
                    else "application/pdf"
                ),
                headers=(
                    {"Content-Disposition": 'attachment; filename="out.pdf"'}
                    if input_filename.endswith((".pdf", ".doc", ".docx"))
                    else None
                ),
            )
        except Exception as e:
            print(f"Error processing file: {e}")
            return Response(
                content="Error while processing the file.",
                status_code=500,
            )
