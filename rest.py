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

    try:
        input_filename = file.filename
        input_filename_ext = input_filename.split(".")[-1]
        tmp_input_file = tempfile.NamedTemporaryFile(
            delete=False, dir=".", suffix=f".{input_filename_ext}"
        )
        tmp_input_file.write(await file.read())
        output_file = tempfile.NamedTemporaryFile(
            delete=False, dir=".", suffix=f".{input_filename_ext}"
        )

        if input_filename.endswith(".pdf"):
            tmp_dir = tempfile.TemporaryDirectory()
            tmp_output_dir = tempfile.TemporaryDirectory()
            pdf_to_images(tmp_input_file.name, tmp_dir.name)
            for img in os.listdir(tmp_dir.name):
                process_image(
                    f"{tmp_dir.name}/{img}",
                    f"{tmp_output_dir.name}/{img}",
                    pd_funcs,
                    lambda img_path: get_image_data(img_path, auto_rotate),
                )
            images_to_pdf(tmp_output_dir.name, output_file.name)
            tmp_dir.cleanup()
            tmp_output_dir.cleanup()
        elif input_filename.endswith((".doc", ".docx")):
            # Create temporary directories for conversion
            tmp_pdf = tempfile.NamedTemporaryFile(delete=False, dir=".", suffix=".pdf")
            tmp_dir = tempfile.TemporaryDirectory()
            tmp_output_dir = tempfile.TemporaryDirectory()
            
            # Convert doc/docx to PDF
            docx_to_pdf(tmp_input_file.name, tmp_pdf.name)
            
            # Convert PDF to images
            pdf_to_images(tmp_pdf.name, tmp_dir.name)
            
            # Process each image
            for img in os.listdir(tmp_dir.name):
                process_image(
                    f"{tmp_dir.name}/{img}",
                    f"{tmp_output_dir.name}/{img}",
                    pd_funcs,
                    lambda img_path: get_image_data(img_path, auto_rotate),
                )
            
            # Convert processed images back to PDF
            images_to_pdf(tmp_output_dir.name, output_file.name)
            
            # Cleanup temporary files
            tmp_pdf.close()
            os.unlink(tmp_pdf.name)
            tmp_dir.cleanup()
            tmp_output_dir.cleanup()
        elif input_filename.endswith((".png", ".jpg", ".jpeg")):
            process_image(
                tmp_input_file.name,
                output_file.name,
                pd_funcs,
                lambda img_path: get_image_data(img_path, auto_rotate),
            )
        else:
            try:
                output_file.close()
                os.unlink(output_file.name)
                tmp_input_file.close()
                os.unlink(tmp_input_file.name)
            except:
                pass
            return Response(
                content="Unsupported file format. Please upload a .pdf, .png, .jpg, .jpeg, .doc, or .docx file.",
                status_code=415,
            )

        output_content = output_file.read()
        output_file.close()
        os.unlink(output_file.name)
        tmp_input_file.close()
        os.unlink(tmp_input_file.name)

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
        print(e)
        try:
            output_file.close()
            os.unlink(output_file.name)
            tmp_input_file.close()
            os.unlink(tmp_input_file.name)
        except:
            pass
        return Response(
            content="Error while processing the file.",
            status_code=500,
        )
