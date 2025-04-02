from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import os
import zipfile

app = FastAPI()

# 🧪 Test endpoint
@app.get("/")
def root():
    return {"message": "✅ Domates API çalışıyor!"}

# Model ve tokenizer için global değişkenler
tokenizer = None
session = None

# Dosya yolları
MODEL_ZIP_PATH = "bert_model.zip"
MODEL_PATH = "bert_domates_model_quant.onnx"

@app.on_event("startup")
def startup_event():
    global tokenizer, session

    # Eğer .onnx dosyası yoksa zip içinden çıkar
    if not os.path.exists(MODEL_PATH):
        print("📦 Zip dosyasından model çıkarılıyor...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Model çıkarıldı.")

    # Tokenizer Hugging Face üzerinden
    tokenizer = AutoTokenizer.from_pretrained("Kahsi13/DomatesRailway")

    # ONNX modeli yükle
    session = onnxruntime.InferenceSession(MODEL_PATH)
    print("✅ Tokenizer ve model yüklendi.")

# Kullanıcıdan gelen metin yapısı
class InputText(BaseModel):
    text: str

# Tahmin endpoint'i
@app.post("/predict")
def predict(input: InputText):
    try:
        if tokenizer is None or session is None:
            return {"error": "⏳ Model yükleniyor, lütfen birazdan tekrar deneyin."}

        encoding = tokenizer.encode_plus(
            input.text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        ort_outs = session.run(None, ort_inputs)
        prediction = int(np.argmax(ort_outs[0]))

        return {"prediction": prediction}
    
    except Exception as e:
        return {"error": str(e)}
