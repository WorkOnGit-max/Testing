"""
Converts sentiment_model.keras → sentiment_model.tflite
Run this ONCE locally before deploying to Render.
"""
import tensorflow as tf
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "BackEnd" / "model"

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_DIR / "sentiment_model.keras", compile=False)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # quantize → smaller + faster
tflite_model = converter.convert()

out_path = MODEL_DIR / "sentiment_model.tflite"
out_path.write_bytes(tflite_model)
print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
