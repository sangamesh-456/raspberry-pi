#!/usr/bin/env bash
# One-time setup on the Raspberry Pi:   cd ~/Downloads/raspberry-pi && bash setup.sh
set -e
cd "$(dirname "$0")"
if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
  echo "installing python3-venv (needs sudo)"; sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
fi
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install numpy opencv-python-headless onnxruntime
if ! ./venv/bin/pip install ai-edge-litert; then
  echo "ai-edge-litert not available for this Python - trying tflite-runtime"
  ./venv/bin/pip install tflite-runtime || echo "WARNING: no TFLite runtime. Stump/player detection and ball speed will not run."
fi
mkdir -p output images videos
chmod +x *.py run.sh 2>/dev/null || true
./venv/bin/python check_setup.py
echo
echo "Done. Run a model with:   ./run.sh run_ball_tracking.py"
