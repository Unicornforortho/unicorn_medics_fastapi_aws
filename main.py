import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from math import exp

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

ankle_one = tf.keras.models.load_model('./models/ankle_best_run.h5')

strToModel = {
  "ankle_one": ankle_one
}

@app.post("/predict")
async def predict(modelName: str = Form(...), file: UploadFile = File(...)):
  try:
    model = strToModel[modelName]
    test_data = load_image_into_numpy_array(await file.read())
    result = model.predict(test_data)
    result = result.reshape(4,)
    label = np.argmax(result)
    confidence = softmax(result.tolist(), label)
    return {
      "result": str(label),
      "confidence": str(confidence)
    }
  except KeyError as k:
    return {
      "error": True,
      "message": "Model not found, please try again later!",
      "modelName": k
    }

def softmax(vector: list, label: int):
  e = []
  print(vector)
  for i in vector:
    e.append(exp(i))
  return e[label] / sum(e)

def load_image_into_numpy_array(data):
  npimg = np.frombuffer(data, np.uint8)
  frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image = cv2.resize(frame, (224, 224))
  test_data = np.array(image).reshape(1, 224, 224, 3)
  return test_data