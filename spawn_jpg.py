import os
import subprocess
import torch
import multiprocessing as mp
import glob

def process_gpu(gpu_id, assigned_tasks):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    if not assigned_tasks:
        print(f"[GPU {gpu_id}] No tasks assigned.", flush=True)
        return
        
    print(f"[GPU {gpu_id}] Assigned {len(assigned_tasks)} frame directories.", flush=True)
    
    # Point directly to the virtual environment binary to avoid uv locking
    tracklab_bin = os.path.abspath("/workspace/sn-gamestate/.venv/bin/tracklab")
    if not os.path.exists(tracklab_bin):
        tracklab_bin = "tracklab" # Fallback if .venv is located elsewhere
        
    for frame_dir_name, frames_path in assigned_tasks:
        print(f"\nProcessing frame directory {frame_dir_name} on GPU {gpu_id}", flush=True)
        # Updated: parent directory of frames_path is now directly the match_name folder
        match_name = os.path.basename(os.path.dirname(frames_path))
        run_name = f"{match_name}_{frame_dir_name}"

        cmd = [
            tracklab_bin, "-cn", "soccernet",
            "dataset=image_folder",
            f"dataset.video_path={frames_path}",
            "dataset.eval_set=val",
            "eval_tracking=false",
            "visualization=none",
            "pipeline=[bbox_detector,reid,track,pitch,calibration,tracklet_agg,team,team_side]",
            "modules.tracklet_agg.cfg.attributes=[role]",
            "modules.calibration.image_width=1024",
            "modules.calibration.image_height=576",
            f"state.save_file=states/sn-gamestate_{frame_dir_name}.pklz",
            f"hydra.run.dir=outputs/sn-gamestate/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}_{run_name}",
        ]
        
        subprocess.check_call(cmd, env=env)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    bds = glob.glob("*.tar.gz")
    bds_cleaned = [bd[:-7] for bd in bds]

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.", flush=True)
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected by PyTorch.")

    os.makedirs("videos", exist_ok=True)
    for bd in bds_cleaned:
        str_targz = bd + ".tar.gz"
        target_match_dir = os.path.join("videos", bd)
        # Skip extracting again if the folder was already moved into videos/
        if not os.path.exists(target_match_dir):
            subprocess.check_call(["tar", "xvzf", str_targz])
            subprocess.check_call(["mv", bd, "videos/"])

    base_dir = "videos"
    if not os.path.exists(base_dir):
        raise RuntimeError(f"Directory '{base_dir}' not found.")
        
    # 1. Collect ALL individual frame directory tasks globally
    match_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    all_tasks = []
    
    for match_name in match_dirs:
        match_dir = os.path.join(base_dir, match_name)
        
        # Look directly inside videos/<match_name>/ for subdirectories
        frame_dirs = sorted([
            d for d in os.listdir(match_dir)
            if os.path.isdir(os.path.join(match_dir, d))
        ])
        for frame_dir_name in frame_dirs:
            frames_path = os.path.abspath(os.path.join(match_dir, frame_dir_name))
            all_tasks.append((frame_dir_name, frames_path))
            
    print(f"Total frame directories found: {len(all_tasks)}", flush=True)
    
    # 2. Distribute tasks round-robin across GPUs
    gpu_tasks = [[] for _ in range(num_gpus)]
    for idx, task in enumerate(all_tasks):
        gpu_id = idx % num_gpus
        gpu_tasks[gpu_id].append(task)
        
    # 3. Spawn processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=process_gpu, args=(gpu_id, gpu_tasks[gpu_id]))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    failed_processes = [
        f"GPU {gpu_id} (exit code {p.exitcode})"
        for gpu_id, p in enumerate(processes)
        if p.exitcode != 0
    ]
    if failed_processes:
        raise RuntimeError("GPU workers failed: " + ", ".join(failed_processes))