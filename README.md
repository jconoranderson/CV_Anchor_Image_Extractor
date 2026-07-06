# Video Face Anchor Extractor

A high-performance Python pipeline that automatically extracts unique individuals from video footage (like security cameras) and saves the sharpest anchor images for each person.

## Features
- **OpenCV DNN Face Detection:** Uses OpenCV's Deep Learning Fully Convolutional Network to robustly detect tiny or poorly angled faces in native resolution.
- **Face Encoding & Clustering:** Uses `face_recognition` to extract 128D facial embeddings and groups them by unique individual using DBSCAN clustering.
- **Sharpness Filtering:** Automatically evaluates the image quality using Laplacian variance to output only the absolute sharpest, clearest frames of each person.
- **Performance Optimized:** Uses native OpenCV video frame jumping (`CAP_PROP_POS_FRAMES`) to instantly skip decoding unneeded frames, reducing processing time from hours to seconds.

## Usage

Place any videos (`.mp4`, `.mov`, `.avi`) into the `input_videos/` directory and run the script:

```bash
python3 extract_cv2_anchors.py
```

The script will automatically cluster the faces and save the top 5 sharpest frames for each unique person into the `anchor_images/` directory.

## Configuration

You can easily adjust the settings at the top of `extract_cv2_anchors.py`:
- `TOP_N_PER_PERSON`: Number of images to output for each unique individual (default: 5).
- `SAMPLE_RATE`: The frame skip interval (default: 30).
- `EPS`: The strictness for grouping people together in DBSCAN.
