from fastapi import FastAPI, File, Response, UploadFile
from solution import process_image
from parse import *

app = FastAPI()

@app.get("/")
async def read_root():
    return "Сервис для анонимизации документов. Загрузите изображение с текстом и получите его с анонимизированными персональными данными."

# upload image and return modified file
@app.post(
    "/upload/",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def upload_image(file: UploadFile = File(...)) -> Response:
    # use tmp file to store image
    with open("tmp.png", "wb") as buffer:
        buffer.write(file.file.read())
    process_image(
        file.filename,
        "output.png",
        [get_all_names_mystem, get_all_addresses_natasha, get_phone_numbers_positions],
    )
    return Response(content=open("output.png", "rb").read(), media_type="image/png")
