from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = 'ai4bharat/IndicBERTv2-MLM-Sam-TLM'

print(f'Downloading and exporting {model_id} to ONNX...')
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
model.save_pretrained('/build/model')
AutoTokenizer.from_pretrained(model_id).save_pretrained('/build/model')
print('ONNX export complete')