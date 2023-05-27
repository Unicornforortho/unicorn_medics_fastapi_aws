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

ankle = tf.keras.models.load_model('./models/ankle.h5', compile=False)
ankle2 = tf.keras.models.load_model('./models/ankle2.h5', compile=False)
shoulder_reverse = tf.keras.models.load_model('./models/shoulder_reverse.h5', compile=False)
shoulder_total = tf.keras.models.load_model('./models/shoulder_total.h5', compile=False)
knee = tf.keras.models.load_model('./models/knee.h5', compile=False)
knee2 = tf.keras.models.load_model('./models/knee2.h5', compile=False)
wrist = tf.keras.models.load_model('./models/wrist.h5', compile=False)

strToModel = {
  "ankle": ankle,
  "ankle2": ankle2,
  "shoulder_reverse": shoulder_reverse,
  "shoulder_total": shoulder_total,
  "knee": knee,
  "knee2": knee2,
  "wrist": wrist,
}

predictionToLink = {
  "ankle": {
    "0": {
      "name": "Depuy Mobility",
      "link": "http://www.rpa.spot.pt/getdoc/5243102b-0def-4108-a82f-8b59568ad3b7/Mobility-Total-Ankle-Surgical-Technique.aspx"
    },
    "1": {
      "name": "Stryker Star",
      "link": "https://www.stryker.com/kw/en/foot-and-ankle/products/star/index-eu.html"
    },
    "2": {
      "name": "Wright Inbone II",
      "link": "https://www.wright.com/footandankleproducts/inbone-ii-total-ankle-system"
    },
    "3": {
      "name": "Zimmer Biomet Trabecular Model",
      "link": "https://www.zimmerbiomet.com/content/dam/zimmer-biomet-OUS-Surg-techniques/foot-and-ankle/trabecular-metal-total-ankle-surgical-technique.pdf"
    },
  },
  "ankle2": {
    "0": {
      "name": "Depuy Agility",
      "link": "https://pubmed.ncbi.nlm.nih.gov/18692011/"
    },
    "1": {
      "name": "Integra Hintegra",
      "link": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8023928/"
    },
    "2": {
      "name": "Tornier Salto",
      "link": "https://www.londonfootankle.co.uk/downloads/SALTO-technique.pdf"
    },
    "3": {
      "name": "Wright Infinity",
      "link": "https://www.wright.com/footandankleproducts/infinity-total-ankle-system"
    },
  },
  "shoulder_reverse": {
    "0": {
      "name": "Depuy Delta Xtend",
      "link": "http://synthes.vo.llnwd.net/o16/LLNWMB8/INT%20Mobile/Synthes%20International/Product%20Support%20Material/legacy_Synthes_PDF/103333.pdf"
    },
    "1": {
      "name": "Evolutis Unic",
      "link": "https://link.springer.com/chapter/10.1007/978-3-319-20840-4_36"
    },
  },
  "shoulder_total": {
    "0": {
      "name": "Bigliani",
      "link": "https://www.zimmerbiomet.com/content/dam/zimmer-biomet-OUS-Surg-techniques/shoulder/bigliani-flatow-surgical-technique.pdf"
    },
    "1": {
      "name": "BioModular",
      "link": "http://faculty.washington.edu/alexbert/Shoulder/Surgery/BiometBioModularChoiceShoulderSystem.pdf"
    },
    "2": {
      "name": "CofieldII",
      "link": "https://www.cofield-ii.com/"
    },
    "3": {
      "name": "Depuy Global",
      "link": "https://www.jnjmedtech.com/en-EMEA/product/global-unite-shoulder-system"
    },
    "4": {
      "name": "Depuy Global Advantage",
      "link": "https://www.camaramx.com/wp-content/themes/camara/images/pdf/16-Global-Advantage.pdf"
    },
    "5": {
      "name": "Global Fracture",
      "link": "https://www.jnjmedtech.com/en-EMEA/product/global-unite-shoulder-system"
    },
    "6": {
      "name": "HRP",
      "link": "http://shoulderarthritis.blogspot.com/2016/10/fixation-of-humeral-component-with.html"
    },
  },
  "knee": {
    "0": {
      "name": "Depuy Attune",
      "link": "https://www.jnjmedtech.com/en-US/product/attune-knee-system"
    },
    "1": {
      "name": "DJO 3D Knee",
      "link": "https://www.djoglobal.com/products/surgical/empowr-3d-knee"
    },
    "2": {
      "name": "Link Gemini SL",
      "link": "http://www.linkorthopaedics.in/us/for-the-physician/products/knee/totalknee.html"
    },
    "3": {
      "name": "Microport Medialpivot",
      "link": "https://www.businesswire.com/news/home/20180130005228/en/Study-Shows-Patients-with-MicroPort%E2%80%99s-Medial-Pivot-Knee-Are-More-Likely-to-Forget-They%E2%80%99ve-Had-a-Joint-Replacement"
    },
    "4": {
      "name": "Zimmer LPS Flex Knee GSF",
      "link": "http://surgitech.net/wp-content/uploads/2018/11/Gender_Solutions_NexGen_High-Flex_Implants_Brochure_97-5764-001-00_10_2006.pdf"
    },
  },
  "knee2": {
    "0": {
      "name": "Exatech Opterak",
      "link": "https://au.exac.com/knee/optetrak-logic-primary-system/"
    },
    "1": {
      "name": "Smith Legion",
      "link": "https://smith-nephew.com/en-us/health-care-professionals/products/orthopaedics/legion-tks"
    },
    "2": {
      "name": "Stryker NRG",
      "link": "https://www.strykermeded.com/media/1502/scorpio-nrg-ar-technique-lsnrgar_st.pdf"
    },
    "3": {
      "name": "Zimmer LPS",
      "link": "https://www.zimmerbiomet.com/content/dam/zimmer-biomet-OUS-Surg-techniques/knee/1393.1-GLBL-en%20NexGen%20LPS%20Fixed%20Bearing%20Knee%20SurgTech-DIGITAL.pdf"
    },
    "4": {
      "name": "Zimmer Persona",
      "link": "https://www.zimmerbiomet.com/en/products-and-solutions/specialties/knee/persona-knee-system.html"
    },
  },
  "wrist": {
    "0": {
      "name": "Depuy Biax",
      "link": "https://www.researchgate.net/publication/51205490_Results_of_189_wrist_replacements_A_report_from_the_Norwegian_Arthroplasty_Register"
    },
    "1": {
      "name": "Integra Universal 2",
      "link": "http://fischermedical.dk/wp-content/uploads/Integra_Universal2_SurgicalTehnique.pdf"
    },
    "2": {
      "name": "Zimmer Biomet Maestro",
      "link": "https://www.zimmerbiomet.com/content/zimmerbiomet/master/zb-latam-master/en/medical-professionals/trauma/product/maestro-wrist-reconstructive-system.html"
    },
  }
}

@app.post("/predict")
async def predict(modelName: str = Form(...), file: UploadFile = File(...)):
  try:
    if validate_model_name(modelName):
      model = strToModel[modelName]
      test_data = load_image_into_numpy_array(await file.read(), modelName)
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

def load_image_into_numpy_array(data, modelName):
  npimg = np.frombuffer(data, np.uint8)
  frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image = cv2.resize(frame, (224, 224))
  test_data = np.array(image).reshape(1, 224, 224, 3)
  if modelName == "shoulder_total" or modelName == "wrist" or modelName == "knee2":
    test_data = test_data / 255.0
  return test_data

def validate_model_name(modelName):
  return modelName in strToModel.keys()