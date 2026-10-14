import os
import subprocess

def main():
    base_dir = "videos"
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found.")
        return
        
    # Walk through the directory structure
    for match_name in sorted(os.listdir(base_dir)):
        match_path = os.path.join(base_dir, match_name)
        if not os.path.isdir(match_path):
            continue
            
        videos_dir = os.path.join(match_path, "videos")
        if not os.path.exists(videos_dir):
            continue
            
        mp4_files = sorted([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])
        
        for filename in mp4_files:
            video_path = os.path.join(videos_dir, filename)
            video_name_no_ext = os.path.splitext(filename)[0]
            
            # Define output directory for frames (e.g., inside a 'frames' folder next to the video)
            output_frames_dir = os.path.join(videos_dir, f"{video_name_no_ext}_frames")
            os.makedirs(output_frames_dir, exist_ok=True)
            
            print(f"Extracting frames for {filename}...")
            
            # Note: Modern ffmpeg uses -fps_mode passthrough instead of the deprecated -vsync 0
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-q:v", "2", "-fps_mode", "passthrough",
                os.path.join(output_frames_dir, "frame_%06d.jpg")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error on {filename}: {result.stderr}")
            else:
                print(f"Finished {filename} -> {output_frames_dir}")

if __name__ == "__main__":
    main()
