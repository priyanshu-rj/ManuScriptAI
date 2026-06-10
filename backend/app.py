from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from PIL import Image

import cv2
import numpy as np
import torch
import joblib





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





print("Loading TrOCR...")

processor = TrOCRProcessor.from_pretrained(
    "./model/trocr_tridis_model"
)

trocr = VisionEncoderDecoderModel.from_pretrained(
    "./model/trocr_tridis_model"
)

print("Loading Random Forest...")

language_model = joblib.load(
    "./model/language_rf.pkl"
)

print("Loading XGBoost...")

script_model = joblib.load(
    "./model/script_xgb.pkl"
)

script_encoder = joblib.load(
    "./model/script_encoder.pkl"
)



century_model = joblib.load(
    "./model/century_xgb.pkl"
)

century_encoder = joblib.load(
    "./model/century_encoder.pkl"
)

# model century 
# century_model = joblib.load(
#     "./model/century_xgb.pkl"
# )

# century_encoder = joblib.load(
#     "./model/century_encoder.pkl"
# )

# century_vectorizer = joblib.load(
#     "./model/century_vectorizer.pkl"
# )



device = "cuda" if torch.cuda.is_available() else "cpu"

trocr.to(device)
trocr.eval()

print("All Models Loaded!")



@app.get("/")
def home():
    return {
        "message": "Cultural Heritage AI Backend Running"
    }



@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    try:

      

        image_bytes = await file.read()

        nparr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return {
                "success": False,
                "error": "Invalid image"
            }



        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        denoise = cv2.fastNlMeansDenoising(
            gray,
            None,
            10,
            7,
            21
        )

        enhanced = cv2.equalizeHist(
            denoise
        )

        pil_image = Image.fromarray(
            enhanced
        )

       
        # TrOCR
       

        pixel_values = processor(
            pil_image,
            return_tensors="pt"
        ).pixel_values

        pixel_values = pixel_values.to(
            device
        )

        with torch.no_grad():

            generated_ids = trocr.generate(
                pixel_values,
                max_length=128
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        text = text.strip()

       
        # RANDOM FOREST
        

        language = language_model.predict(
            [text]
        )[0]

        
        # XGBOOST
      

        script_id = script_model.predict(
            [text]
        )[0]

        script = script_encoder.inverse_transform(
            [script_id]
        )[0]
     
        century_id = century_model.predict(
           [text]
        )[0]

        century = century_encoder.inverse_transform(
           [century_id]
        )[0]



        return {
          "success": True,
          "ocr_text": text,
          "language": language,
          "script": script,
          "century": century
         }


    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }