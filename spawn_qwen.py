import os
import subprocess
import torch
from multiprocessing import Process

def process_gpu(gpu_id):
    # Set environment for this GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Find all video directories assigned to this GPU
    video_dirs = sorted([d for d in os.listdir() if os.path.isdir(d) and 'videos' in d])
    assigned_dirs = []
    for idx, dir_name in enumerate(video_dirs):
        if idx % torch.cuda.device_count() == gpu_id:
            assigned_dirs.append(dir_name)
    
    # Process each directory's videos sequentially on this GPU
    for dir_name in assigned_dirs:
        videos_dir = os.path.join(dir_name, "videos")
        mp4_files = sorted([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])
        
        for filename in mp4_files:
            video_path = os.path.join(videos_dir, filename)
            print(f"\nProcessing {filename} on GPU {gpu_id}")
            
            # Run your command with the specific video path
            cmd = [
                "uv", "run", "tracklab", "-cn", "soccernet",
                "--dataset", "youtube",
                "--dataset.video_path", video_path,
                "--eval_tracking", "false"
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print(f"Output for {filename}:\n{result.stdout}")
            if result.returncode != 0:
                print(f"Error processing {filename}: {result.stderr}")

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    processes = []
    
    # Launch one process per GPU
    for gpu_id in range(num_gpus):
        p = Process(target=process_gpu, args=(gpu_id,))
        p.start()
        processes.append(p)
    
    # Wait for all GPUs to complete
    for p in processes:
        p.join()
