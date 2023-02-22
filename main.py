import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

model = tf.keras.models.load_model('./models/ankle_best_run.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  test_data = load_image_into_numpy_array(await file.read())
  result = model.predict(test_data)
  result = result.reshape(4,)
  result = result * 100
  total_sum = sum(result)
  label = np.argmax(result)
  confidence = float(result[label].round(2))/total_sum
  return {
    "result": str(label),
    "confidence": str(confidence)
  }

def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(frame, (224, 224))
    test_data = np.array(image).reshape(1, 224, 224, 3)
    return test_data