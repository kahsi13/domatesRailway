from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import requests
import os

app = FastAPI()

# 🧪 Test endpoint
@app.get("/")
def root():
    return {"message": "✅ Domates API çalışıyor!"}

# Model ve Tokenizer global tanım
tokenizer = None
session = None

MODEL_URL = "https://huggingface.co/Kahsi13/DomatesRailway/resolve/main/bert_domates_model_quant.onnx"
MODEL_PATH = "bert_domates_model_quant.onnx"


# 🚀 Startup'ta model ve tokenizer yüklenir
@app.on_event("startup")
def startup_event():
    global tokenizer, session

    if not os.path.exists(MODEL_PATH):
        print("🔽 Model indiriliyor...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model indirildi!")

    tokenizer = AutoTokenizer.from_pretrained("Kahsi13/DomatesRailway")
    session = onnxruntime.InferenceSession(MODEL_PATH)
    print("✅ Tokenizer ve ONNX model yüklendi.")

# 📩 Kullanıcıdan gelen metin yapısı
class InputText(BaseModel):
    text: str

# 🧠 Tahmin endpoint'i
@app.post("/predict")
def predict(input: InputText):
    try:
        if tokenizer is None or session is None:
            return {"error": "⏳ Model henüz hazır değil, lütfen birkaç saniye sonra tekrar deneyin."}

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
