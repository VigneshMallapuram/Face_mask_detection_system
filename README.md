# Face Mask Detection with Live Alert System

This project trains a small grayscale Keras CNN and uses OpenCV Haar Cascades
to detect faces in a webcam stream. Faces classified as `No Mask` receive a
red box and an `ALERT: MASK REQUIRED` overlay.

The detector uses conservative Haar-Cascade settings (`minNeighbors=8` and a
70-pixel minimum face size) to reduce background false positives. Improve
lighting and move closer to the camera if a real face is too small to detect.
Haar Cascades can still detect face-like posters or pictures as faces.
Predictions below 75% confidence are shown as `Uncertain` and do not trigger
the alert. This avoids treating a weak, collapsed model as a reliable
no-mask result.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put training images in this layout. Folder names become the class labels:

```text
data/
  mask/
  no_mask/
```

Train and run:

```powershell
python train.py --data-dir data
python app.py
```

Training prints validation accuracy and refuses to save a model below 70%.
Always retrain after changing or reorganizing the dataset.

Open <http://127.0.0.1:5000>. Set `MASK_MODEL` to use another `.keras` model.
The model must accept 100x100 grayscale images and output either a sigmoid
probability (the training script's format) or two class probabilities.

The webcam is opened only when `/video_feed` is requested. Camera failures and
missing model files are shown as explicit startup/stream errors.
On Windows, the app tries DirectShow before Media Foundation because it is more
reliable with many USB webcams. Set `CAMERA_INDEX=1` if the camera is not index
0, and close Teams, Zoom, or other applications that may own the camera.
If the browser shows a broken image, the webcam stream failed; it is not a
no-mask result. Close other camera applications, check Windows camera
permissions, or try another `CAMERA_INDEX`.

### TensorFlow DLL error on Windows

If TensorFlow fails with `DLL load failed ... Application Control policy has
blocked this file`, Windows is preventing TensorFlow's native binaries from
loading. This is not caused by the dataset or application code. The current
environment must use a Python version supported by the installed TensorFlow
release, and the administrator may need to allow the TensorFlow DLLs. Create a
fresh virtual environment with that supported Python version, then reinstall
`requirements.txt`. The app displays this diagnosis instead of crashing.
