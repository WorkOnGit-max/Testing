"""
Converts sentiment_model.keras → sentiment_model.onnx
Run this ONCE locally (requires tensorflow + tf2onnx installed).

Install conversion deps:
    pip install tf2onnx

Then run:
    python convert_to_onnx.py
"""
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "BackEnd" / "model"
INPUT  = MODEL_DIR / "sentiment_model.keras"
OUTPUT = MODEL_DIR / "sentiment_model.onnx"

print(f"Converting {INPUT.name} -> {OUTPUT.name} ...")

# tf2onnx CLI is the most reliable way for Keras models
result = subprocess.run(
    [
        sys.executable, "-m", "tf2onnx.convert",
        "--keras", str(INPUT),
        "--output", str(OUTPUT),
        "--opset", "13",
    ],
    capture_output=False,
)

if result.returncode == 0:
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\n✅ Saved: {OUTPUT}  ({size_mb:.1f} MB)")
else:
    print("\n❌ Conversion failed — check output above.")
    sys.exit(1)
