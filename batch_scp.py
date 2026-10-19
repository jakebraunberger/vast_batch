import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

# Define nodes with explicit GPU counts alongside connection details
NODES = [
    # {"host": "ssh9.vast.ai", "port": 26905, "user": "root", "gpus": 1},
    {"host": "ssh9.vast.ai", "port": 34741, "user": "root", "gpus": 1}
]

def provision_and_run(node, assigned_videos, parallel_streams=6):
    host = node['host']
    port = node['port']
    user = node['user']
    
    # Combine the assigned video archives and the script into a single transfer list
    files_to_sync = assigned_videos + ["spawn_jpg.py"]
    quoted_files = " ".join(shlex.quote(f) for f in files_to_sync)
    
    # 1. LFTP assigned files and script to the remote node in parallel
    # -c: continues/resumes partially transferred files (including those left by rsync -P)
    # -P: uploads N files in parallel over separate SFTP streams
    # -O: specifies the remote target directory
    lftp_commands = (
        "set sftp:auto-confirm yes; "
        "set sftp:connect-program 'ssh -a -x -o StrictHostKeyChecking=no -c aes128-gcm@openssh.com'; "
        "set net:max-retries 3; "
        "set net:reconnect-interval-base 5; "
        f"mput -c -P {parallel_streams} -O /workspace/sn-gamestate/ {quoted_files}; "
        "bye"
    )
    
    lftp_cmd = [
        "lftp",
        "-u", f"{user},",  # Trailing comma forces SSH key/agent auth without password prompt
        "-p", str(port),
        f"sftp://{host}",
        "-e", lftp_commands
    ]
    
    subprocess.run(lftp_cmd, check=True)
    
    # 2. Build remote command to run worker script mapping processes to available GPUs
    remote_script = f"cd /workspace/sn-gamestate && source .venv/bin/activate && python spawn_jpg.py"
    remote_cmd = f"tmux new-session -d -s inference_job '{remote_script} > run.log 2>&1'"
    
    ssh_cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", f"{user}@{host}", remote_cmd]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
    return host, True

# Example Video Sharding Logic
targzs = glob.glob("000*.tar.gz")
total_gpus = sum(node["gpus"] for node in NODES)

# Distribute video chunks proportionally to each node's GPU capacity
chunk_size = len(targzs) // total_gpus
current_idx = 0

with ThreadPoolExecutor(max_workers=15) as executor:
    futures = []
    for node in NODES:
        node_gpu_count = node["gpus"]
        node_targz_amount = chunk_size * node_gpu_count
        assigned_targzs = targzs[current_idx : current_idx + node_targz_amount]
        current_idx += node_targz_amount
        
        futures.append(executor.submit(provision_and_run, node, assigned_targzs))
        
    for future in as_completed(futures):
        host, success = future.result()
        print(f"Node {host} initialized successfully: {success}")