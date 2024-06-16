import os
import tempfile
from fastapi import FastAPI, File, Response, UploadFile
from read import images_to_pdf, pdf_to_images
from solution import process_image
from parse import *

app = FastAPI()


@app.get("/")
async def read_root():
    return "Сервис для анонимизации документов. Загрузите изображение с текстом и получите его с анонимизированными персональными данными."


# upload image and return modified file
@app.post(
    "/upload/",
    responses={200: {"content": {"image/png": {}, "application/pdf": {}}}},
    response_class=Response,
)
async def upload(file: UploadFile = File(...)) -> Response:
    pd_funcs = [
        get_all_names_mystem,
        extract_complex_address_indices,
        get_phone_numbers_positions,
        get_bd_positions,
        get_specific_numbers,
        find_numeric_sequences,
    ]

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
            )
        images_to_pdf(tmp_output_dir.name, output_file.name)
        tmp_dir.cleanup()
        tmp_output_dir.cleanup()
    elif input_filename.endswith((".png", ".jpg", ".jpeg")):
        process_image(
            tmp_input_file.name,
            output_file.name,
            pd_funcs,
        )
    else:
        return Response(
            content="Unsupported file format. Please upload a .pdf, .png, .jpg, or .jpeg file.",
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
    )
