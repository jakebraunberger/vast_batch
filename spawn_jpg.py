import os
import subprocess
import torch
import multiprocessing as mp

def process_gpu(gpu_id, total_gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    base_dir = "videos"
    if not os.path.exists(base_dir):
        print(f"[GPU {gpu_id}] Directory '{base_dir}' not found.", flush=True)
        return
        
    match_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    print(f"[GPU {gpu_id}] Found match dirs: {match_dirs}", flush=True)
    
    assigned_dirs = []
    for idx, match_name in enumerate(match_dirs):
        if idx % total_gpus == gpu_id:
            assigned_dirs.append(match_name)
            
    for match_name in assigned_dirs:
        videos_dir = os.path.join(base_dir, match_name, "videos")
        if not os.path.exists(videos_dir):
            continue
            
        # Find all directories containing extracted JPG frames
        frame_dirs = sorted([d for d in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, d)) and d.endswith('_frames')])
        
        for frame_dir_name in frame_dirs:
            frames_path = os.path.abspath(os.path.join(videos_dir, frame_dir_name))
            print(f"\nProcessing frame directory {frame_dir_name} on GPU {gpu_id}", flush=True)
            
            cmd = [
                "uv", "run", "tracklab", "-cn", "soccernet",
                "dataset=image_folder",
                f"dataset.video_path={frames_path}",
                "dataset.eval_set=val",
                "eval_tracking=false",
                "visualization=none",
                "pipeline=[bbox_detector,reid,track,pitch,calibration,tracklet_agg,team,team_side]",
                "modules.tracklet_agg.cfg.attributes=[role]",
                # "~visualization.visualizers.radar"
            ]
            
            subprocess.check_call(cmd)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.", flush=True)
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected by PyTorch.")
        
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=process_gpu, args=(gpu_id, num_gpus))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()