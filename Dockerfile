FROM python:3.10

WORKDIR /data
COPY . /data

RUN pip install -r requirements.txt
RUN pip install opencv-python

COPY . .

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80" ]