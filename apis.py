import base64

import solve
import cv2
from fastapi import FastAPI,File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def HomePage():
    return """
    <html>
        <head>
            <title>Hello Solvers</title>
        </head>
        <body>
            <h2>Hello Solvers</h2>
            <h1>PuzzlePro - sodoku solver</h1>
        </body>
    </html>
    """


@app.post("/uploadfile")
async def upload_file(file: UploadFile = File(...)):
 img = await solve.solver(file)
 return {"filename": file.filename, "status": 1, "images": img}
