# Video Face Anchor Extractor

A high-performance Python pipeline that automatically extracts unique individuals from video footage (like security cameras) and saves the sharpest anchor images for each person.

## Features
- **OpenCV DNN Face Detection:** Uses OpenCV's Deep Learning Fully Convolutional Network to robustly detect tiny or poorly angled faces in native resolution.
- **Face Encoding & Clustering:** Uses `face_recognition` to extract 128D facial embeddings and groups them by unique individual using DBSCAN clustering.
- **Sharpness Filtering:** Automatically evaluates the image quality using Laplacian variance to output only the absolute sharpest, clearest frames of each person.
- **Performance Optimized:** Uses native OpenCV video frame jumping (`CAP_PROP_POS_FRAMES`) to instantly skip decoding unneeded frames, reducing processing time from hours to seconds.

## Usage

### 1. Batch Video Processing (Recommended)
To recursively process an entire directory (or day) of videos, use the `process_day_videos.py` wrapper script. This script will automatically find all `.avi` files, process them individually, and create a dedicated anchor images folder next to each video file.

```bash
python process_day_videos.py "/path/to/video/directory"
```
*(Tip: Add `PYTHONUNBUFFERED=1` before the command if you are piping the output to a log file and want live terminal updates).*

### 2. Single Video/Directory Extraction
If you want to process a specific set of videos manually, you can run the core extractor directly. By default, it looks in `input_videos/` and outputs to `anchor_images/`:

```bash
python extract_unique_anchors_fast.py --input_dir /path/to/inputs --output_dir /path/to/outputs
```

## Configuration

You can easily adjust the settings at the top of `extract_unique_anchors_fast.py`:
- `TOP_N_PER_PERSON`: Number of images to output for each unique individual (default: 25).
- `SAMPLE_RATE`: The frame skip interval (default: 30).
- `EPS`: The strictness for grouping people together in DBSCAN.
