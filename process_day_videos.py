import os
import glob
import time
import subprocess
import shutil
import argparse
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTOR_SCRIPT = os.path.join(SCRIPT_DIR, "extract_unique_anchors_fast.py")

def process_videos(base_dir):
    print(f"Searching for videos in {base_dir}")
    
    # 1. Find all .avi files recursively
    search_pattern = os.path.join(base_dir, '**', '*.avi')
    video_files = glob.glob(search_pattern, recursive=True)
    
    if not video_files:
        video_files = glob.glob(os.path.join(base_dir, '*.avi'))
        
    if not video_files:
        print("No .avi videos found.")
        return
        
    print(f"Found {len(video_files)} video files.")
    
    # Process each video individually
    start_time = time.time()
    
    for v_path in video_files:
        v_name_no_ext = os.path.splitext(os.path.basename(v_path))[0]
        v_dir = os.path.dirname(v_path)
        
        # Output directory in the exact same directory as the video, named after the video
        target_output_dir = os.path.join(v_dir, v_name_no_ext)
        
        print(f"\n--- Processing Video: {v_name_no_ext} ---")
        
        # Create a temporary input directory for this video
        temp_input_dir = tempfile.mkdtemp(prefix="video_temp_")
        
        # Symlink file into temp dir
        symlink_path = os.path.join(temp_input_dir, os.path.basename(v_path))
        try:
            os.symlink(v_path, symlink_path)
        except OSError as e:
            print(f"Could not create symlink for {v_path}: {e}")
            print("Copying file instead...")
            shutil.copy2(v_path, symlink_path)
            
        # Run extractor script
        cmd = [
            sys.executable, EXTRACTOR_SCRIPT,
            "--input_dir", temp_input_dir,
            "--output_dir", target_output_dir
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully created anchor images in {target_output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {v_name_no_ext}: {e}")
            
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_input_dir)
        except Exception as e:
            print(f"Failed to clean up {temp_input_dir}: {e}")
        
    end_time = time.time()
    total_time = end_time - start_time
    
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print(f"\nAll videos processed. Total execution time: {int(h)}h {int(m)}m {int(s)}s ({total_time:.2f} seconds).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and extract anchor images grouped by camera.")
    parser.add_argument("directory", help="The base directory to search for videos.")
    args = parser.parse_args()
    
    process_videos(args.directory)