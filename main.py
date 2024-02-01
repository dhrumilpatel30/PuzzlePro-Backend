import json
import secrets

import keras.models
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import io
import base64
from PIL import Image
import numpy as np
from pydantic import BaseModel

from recogniser import recognise_sudoku

app = FastAPI()
security = HTTPBasic()

USERNAME = 'puzzleProAdmin'
PASSWORD = "willBeChangedOnDeployment"

digit_recognition_model = keras.models.load_model('Models/printed_digits_model.keras')


def base64_to_img(base64_data):
    try:
        base64_data = base64_data.split(',')[1]

        # Decode base64 and convert to numpy array
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)

        return image_array
    except Exception as e:
        raise ValueError("Error decoding base64 image data: {}".format(str(e)))


def image_to_matrix(image):
    sudoku_matrix = recognise_sudoku(image, digit_recognition_model)
    return sudoku_matrix


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


class Item(BaseModel):
    base64_image: str


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.post("/generate-sudoku-matrix")
async def add_image_to_matrix(item: Item, authenticated: bool = Depends(authenticate)):
    try:
        image = base64_to_img(item.base64_image)
        matrix = image_to_matrix(image)
        matrix_list = np.array(matrix, dtype=np.int64).tolist()
        return {"matrix": json.dumps(matrix_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
