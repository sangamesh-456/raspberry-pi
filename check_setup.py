#!/usr/bin/env python3
"""Check the Pi is ready: Python packages, every model file, and that each model loads.

    python3 check_setup.py
"""
import os
import platform
import sys
import time

ok = True
print(f"python {platform.python_version()} on {platform.machine()} ({platform.platform()})")
for mod in ("numpy", "cv2", "onnxruntime"):
    try:
        m = __import__(mod); print(f"  [ok] {mod} {getattr(m, '__version__', '')}")
    except ImportError:
        ok = False; print(f"  [MISSING] {mod}  -> run ./setup.sh")
tfl = None
for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
    try:
        __import__(mod); tfl = mod; break
    except ImportError:
        pass
print(f"  [{'ok' if tfl else 'MISSING'}] tflite runtime {tfl or '-> stump/player detection and ball speed will not run'}")
if not ok:
    sys.exit(1)

from crickcat import core as C   # noqa: E402

print(f"\nmodels in {C.MODELS}")
for f in sorted(os.listdir(C.MODELS)):
    p = os.path.join(C.MODELS, f); mb = os.path.getsize(p) / 1e6
    t = time.perf_counter()
    try:
        if f.endswith(".onnx"):
            s = C.onnx(f)
            io = f"in {[tuple(i.shape) for i in s.get_inputs()]} out {[tuple(o.shape) for o in s.get_outputs()]}"
        elif f.endswith(".tflite") and tfl:
            it = C.tflite(f)
            io = f"in {[int(v) for v in it.get_input_details()[0]['shape']]} out {[[int(v) for v in o['shape']] for o in it.get_output_details()]}"
        else:
            io = "skipped (no tflite runtime)"
        print(f"  [ok] {f:30s} {mb:5.1f} MB  load {(time.perf_counter() - t) * 1000:5.0f} ms  {io}")
    except Exception as e:                                   # noqa: BLE001
        ok = False; print(f"  [FAIL] {f}: {e}")
print("\nREADY" if ok else "\nNOT READY - see the lines above")
