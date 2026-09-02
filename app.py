import os
import pickle
import re
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from BackEnd.TSA_image import extract
import psutil

# Minimal preprocessing: no lemmatization or stopwords to save memory
lemmatizer = None
stop_words = set()


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "BackEnd" / "model"
UPLOAD_FOLDER = BASE_DIR / "Uploaded images"
MAX_LEN = 80
LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "FrontEnd" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "FrontEnd" / "templates")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

class SentimentPredictor:
    def __init__(self, model_dir, load_per_request: bool = False):
        self.model_dir = model_dir
        self._session = None
        self._tokenizer = None
        self.load_per_request = load_per_request

    def _load(self):
        if self._session is None:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            # Reduce memory usage settings
            sess_options.enable_mem_pattern = False
            sess_options.enable_cpu_mem_arena = False
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            model_path = self.model_dir / "sentiment_model_quant.onnx"
            if not model_path.exists():
                model_path = self.model_dir / "sentiment_model.onnx"
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        if self._tokenizer is None:
            import tensorflow as tf
            import sys, tensorflow as tf
            sys.modules['keras'] = tf.keras
            with open(self.model_dir / "tokenizer.pkl", "rb") as file:
                self._tokenizer = pickle.load(file)

    def _check_memory(self):
        # Safety margin 440 MB (allow up to 450 MB total)
        proc = psutil.Process()
        rss = proc.memory_info().rss
        if rss > 440 * 1024 * 1024:
            raise MemoryError(f"RAM usage exceeded safe limit: {rss / (1024*1024):.1f} MB")

    def predict(self, text: str) -> tuple[str, float]:
        self._load()
        # Verify memory usage before inference
        self._check_memory()
        sequence = self._tokenizer.texts_to_sequences([text])
        # Pure-numpy pad_sequences (no keras/TF needed)
        seq = sequence[0][:MAX_LEN]
        padded = np.zeros((1, MAX_LEN), dtype="float32")
        padded[0, : len(seq)] = seq
        input_name = self._session.get_inputs()[0].name
        probs = self._session.run(None, {input_name: padded})[0][0]
        label_idx = int(np.argmax(probs))
        result = (LABEL_NAMES[label_idx], float(probs[label_idx]))
        # If loading per request, free resources immediately
        if self.load_per_request:
            self._session = None
            self._tokenizer = None
            import gc
            gc.collect()
        return result

# Global predictor removed; per-request predictor used

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    if lemmatizer is not None:
        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words
        ]

    return " ".join(tokens)

# Configuration: set to True to load model per request (lower peak RAM, higher latency)
LOAD_PER_REQUEST = True

# Initialize shared predictor only when per-request loading is disabled
if not LOAD_PER_REQUEST:
    predictor = SentimentPredictor(MODEL_DIR)

def prediction_result(user_input: str) -> tuple[str, float, str]:
    cleaned = clean_text(user_input)
    if not cleaned:
        return "No readable text", 0.0, ""
    # Choose predictor based on configuration
    if LOAD_PER_REQUEST:
        # Create a fresh predictor for this request and discard after use
        temp_predictor = SentimentPredictor(MODEL_DIR, load_per_request=True)
        result_label, confidence = temp_predictor.predict(cleaned)
        # Explicitly delete to free memory
        del temp_predictor
    else:
        result_label, confidence = predictor.predict(cleaned)
    return result_label, confidence, cleaned


def render_result(request: Request, source_text: str):
    result, confidence, cleaned_text = prediction_result(source_text)
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "result": f"{result} ({confidence:.2%})" if confidence else result,
            "input_text": source_text,
            "cleaned_text": cleaned_text,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/predict", response_class=HTMLResponse)
async def predict_text(request: Request, text: str = Form(...)):
    if not text.strip():
        return HTMLResponse(content="No input provided", status_code=400)

    return render_result(request, text)


@app.post("/upload_image", response_class=HTMLResponse)
async def predict_image(request: Request, image: UploadFile = File(...)):
    if not image.filename:
        return HTMLResponse(content="No image uploaded", status_code=400)

    image_path = UPLOAD_FOLDER / Path(image.filename).name
    with open(image_path, "wb") as file:
        file.write(await image.read())

    extracted_text = extract(str(image_path))
    if not extracted_text:
        return HTMLResponse(content="Could not extract text from image", status_code=400)

    return render_result(request, extracted_text)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    print(f"Open http://127.0.0.1:{port} in your browser")
    uvicorn.run(app, host=host, port=port)
