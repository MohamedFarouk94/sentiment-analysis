from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
import torch

app = FastAPI()

MODEL_DIR = "bert_sentiment"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=1, state_dict=load_file(f"{MODEL_DIR}/model.safetensors")
)
model.eval()

# Load HTML template
with open("static/index.html", "r") as f:
    html_template = f.read()


@app.get("/", response_class=HTMLResponse)
async def home():
    return html_template


@app.post("/predict/")
async def predict(text: str = Form(...)):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        logit = model(**inputs).logits.squeeze()
        probability = torch.sigmoid(logit).item()  # Convert logit to probability (0-1)

    sentiment_score = int(probability * 100)  # Scale to 0-100

    return f"Sentiment Score: {sentiment_score}/100"
