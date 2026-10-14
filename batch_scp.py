import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define nodes with explicit GPU counts alongside connection details
NODES = [
    {"host": "192.168.1.101", "port": 2222, "user": "root", "gpus": 1},
    {"host": "192.168.1.102", "port": 2223, "user": "root", "gpus": 2},
    {"host": "192.168.1.103", "port": 2224, "user": "root", "gpus": 4},
    # ... up to 15 nodes
]

def provision_and_run(node, assigned_videos):
    target = f"{node['user']}@{node['host']}"
    port = node['port']
    
    # 1. SCP assigned video files/chunk manifest to the remote node
    scp_cmd = ["scp", "-P", str(port), "-o", "StrictHostKeyChecking=no"] + assigned_videos + [f"{target}:/workspace/ball/data/"]
    subprocess.run(scp_cmd, check=True, capture_output=True)
    
    # 2. Build remote command to run worker script mapping processes to available GPUs
    remote_script = f"cd /workspace/ball && python -m ball.mine.src.run_batch_workers --num-gpus {node['gpus']}"
    remote_cmd = f"tmux new-session -d -s inference_job '{remote_script} > run.log 2>&1'"
    
    ssh_cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", target, remote_cmd]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
    return node['host'], True

# Example Video Sharding Logic
all_videos = [f"video_{i}.mp4" for i in range(100)]
total_gpus = sum(node["gpus"] for node in NODES)

# Distribute video chunks proportionally to each node's GPU capacity
chunk_size = len(all_videos) // total_gpus
current_idx = 0

with ThreadPoolExecutor(max_workers=15) as executor:
    futures = []
    for node in NODES:
        node_gpu_count = node["gpus"]
        node_video_count = chunk_size * node_gpu_count
        assigned_videos = all_videos[current_idx : current_idx + node_video_count]
        current_idx += node_video_count
        
        futures.append(executor.submit(provision_and_run, node, assigned_videos))
        
    for future in as_completed(futures):
        host, success = future.result()
        print(f"Node {host} initialized successfully: {success}")