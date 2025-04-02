import base64

with open("bert_domates_model_quant.onnx", "rb") as f:
    encoded = base64.b64encode(f.read())

with open("bert_model_base64.txt", "wb") as out:
    out.write(encoded)

print("✅ Encode işlemi tamamlandı. Dosya: bert_model_base64.txt")
