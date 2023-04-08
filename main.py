import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

ankle = tf.keras.models.load_model('./models/ankle.h5')
shoulder_reverse = tf.keras.models.load_model('./models/shoulder_reverse.h5', compile=False)
knee = tf.keras.models.load_model('./models/knee.h5')

strToModel = {
  "ankle": ankle,
  "shoulder_reverse": shoulder_reverse,
  "knee": knee,
}

predictionToLink = {
  "ankle": {
    "0": {
      "name": "Depuy Mobility",
      "link": "https://www.depuy-mobility.com/"
    },
    "1": {
      "name": "Stryker Star",
      "link": "https://www.stryker-star.com/"
    },
    "2": {
      "name": "Wright Inbone II",
      "link": "https://www.write-inbone-ii.com/"
    },
    "3": {
      "name": "Zimmer Biomet Trabecular Model",
      "link": "https://www.zimmer-biomet-trabecular-model.com/"
    },
  },
  "shoulder_reverse": {
    "0": {
      "name": "Depuy Delta Xtend",
      "link": "https://www.depuy-delta-xtend.com/"
    },
    "1": {
      "name": "Evolutis Unic",
      "link": "https://www.evolutis-unic.com/"
    },
  },
  "knee": {
    "0": {
      "name": "Zimmer LPS Flex Knee GSF",
      "link": "https://www.zimmer-lps-flex-knee-gsf.com/"
    },
    "1": {
      "name": "Depuy Attune",
      "link": "https://www.depu-attune.com/"
    },
    "2": {
      "name": "Microport Medialpivot",
      "link": "https://www.microport-medialpivot.com/"
    },
    "3": {
      "name": "Link Gemini SL",
      "link": "https://www.link-gemini-sl.com/"
    },
    "4": {
      "name": "DJO 3D Knee",
      "link": "https://www.djo-3d-knee.com/"
    },
  }
}

@app.post("/predict")
async def predict(modelName: str = Form(...), file: UploadFile = File(...)):
  print(modelName)
  try:
    if validate_model_name(modelName):
      model = strToModel[modelName]
      test_data = load_image_into_numpy_array(await file.read())
      number_of_classes = model.output_shape[1]
      result = model.predict(test_data)
      result = result.reshape(number_of_classes,)
      label = np.argmax(result)
      implantName = predictionToLink[modelName][str(label)]["name"]
      implantLink = predictionToLink[modelName][str(label)]["link"]
      return {
        "result": str(label),
        "implantName": str(implantName),
        "implantLink": str(implantLink)
      }
    else:
      return {
        "error": True,
        "message": "Model not found, please try again later!",
      }
  except KeyError as k:
    return {
      "error": True,
      "message": "Model not found, please try again later!",
      "modelName": k
    }

def load_image_into_numpy_array(data):
  npimg = np.frombuffer(data, np.uint8)
  frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image = cv2.resize(frame, (224, 224))
  test_data = np.array(image).reshape(1, 224, 224, 3)
  return test_data

def validate_model_name(modelName):
  return modelName in strToModel.keys()