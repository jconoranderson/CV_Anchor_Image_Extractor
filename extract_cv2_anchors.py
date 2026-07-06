import cv2
import os
import glob
import face_recognition
import numpy as np
import urllib.request
from sklearn.cluster import DBSCAN

INPUT_DIR = 'input_videos'
OUTPUT_DIR = 'anchor_images'
TOP_N_PER_PERSON = 5  # Number of anchor images to output per unique person
SAMPLE_RATE = 30 # Reduced to 30 (1 second gap). It's fast enough now to handle it!
EPS = 0.6
MIN_SAMPLES = 1

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Download OpenCV DNN Face Detector models if missing ---
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

def download_model_files():
    if not os.path.exists(PROTOTXT_PATH):
        print("Downloading OpenCV face detector prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    if not os.path.exists(MODEL_PATH):
        print("Downloading OpenCV face detector model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def calculate_sharpness(image):
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main():
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV', '*.AVI']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    if not video_files:
        print(f"No videos found in {INPUT_DIR}/")
        return

    # Automatically download the pre-trained Deep Learning models
    download_model_files()
    
    # Initialize OpenCV's Deep Learning Face Detector (Extremely fast and robust, no extra dependencies!)
    # Using readNet instead of readNetFromCaffe for compatibility with OpenCV 5+
    net = cv2.dnn.readNet(MODEL_PATH, PROTOTXT_PATH)

    faces_data = []
    print(f"Extracting faces using OpenCV Deep Learning (Sample Rate: {SAMPLE_RATE})...")

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue
            
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
                
            ih, iw, _ = frame.shape
            
            # Prepare image for DNN (use native resolution to catch tiny faces)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (iw, ih), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            
            locations = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter out weak detections (0.4 confidence)
                if confidence > 0.4:
                    box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
                    x_min, y_min, x_max, y_max = box.astype("int")
                    
                    # Ensure bounding box is within frame dimensions
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(iw, x_max), min(ih, y_max)
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # Ignore tiny background faces (lowered to 20x20 for distant security cameras)
                    if w < 20 or h < 20:
                        continue
                        
                    # face_recognition expects (top, right, bottom, left)
                    locations.append((y_min, x_max, y_max, x_min))
            
            if locations:
                # Convert to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                encodings = face_recognition.face_encodings(rgb_frame, locations)
                
                for loc, encoding in zip(locations, encodings):
                    top, right, bottom, left = loc
                    crop = frame[top:bottom, left:right]
                    sharpness = calculate_sharpness(crop)
                    
                    faces_data.append({
                        'encoding': encoding,
                        'video_path': video_path,
                        'video_name': video_name,
                        'frame_index': frame_count,
                        'sharpness': sharpness
                    })
                print(f"Found {len(locations)} valid faces at frame {frame_count} in {video_name}")
                    
            frame_count += SAMPLE_RATE
            
        cap.release()
        print(f"Finished {video_name}. Total faces extracted so far: {len(faces_data)}")

    if not faces_data:
        print("No faces detected in any videos.")
        return

    print("Clustering faces...")
    encodings_list = [d['encoding'] for d in faces_data]
    
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='euclidean')
    dbscan.fit(encodings_list)
    
    labels = dbscan.labels_
    
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(faces_data[i])
        
    print(f"Found {len(clusters)} unique people (clusters).")
    
    for label, items in clusters.items():
        # Sort faces in this cluster by sharpness descending
        top_items = sorted(items, key=lambda x: x['sharpness'], reverse=True)[:TOP_N_PER_PERSON]
        
        for rank, best_item in enumerate(top_items, start=1):
            cap = cv2.VideoCapture(best_item['video_path'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_item['frame_index'])
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                output_filename = f"person_{label}_rank{rank}_{best_item['video_name']}_frame{best_item['frame_index']}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print(f"Saved {output_filename} (Sharpness: {best_item['sharpness']:.2f})")
            else:
                print(f"Failed to retrieve frame {best_item['frame_index']} from {best_item['video_path']}")

if __name__ == "__main__":
    main()
